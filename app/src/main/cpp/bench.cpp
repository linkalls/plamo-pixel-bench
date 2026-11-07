// PLaMo 3 Translate 0.8B: unified Android APK with three separately compiled CPU engines.
// Each variant is built as a self-contained static-linked llama.cpp JNI library.
// Runtime progress, cancellation and a hard execution deadline prevent indefinite loading.
#include <jni.h>
#include <llama.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static std::atomic<bool> cancelled{false};

static double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

static std::string utf8(JNIEnv *env, jstring s) {
    if (!s) return {};
    const char *chars = env->GetStringUTFChars(s, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(s, chars);
    return result;
}

static void init_llama() {
    static std::once_flag once;
    std::call_once(once, [] { llama_backend_init(); });
}

struct ModelDeleter {
    void operator()(llama_model *p) const { if (p) llama_model_free(p); }
};
struct ContextDeleter {
    void operator()(llama_context *p) const { if (p) llama_free(p); }
};
struct SamplerDeleter {
    void operator()(llama_sampler *p) const { if (p) llama_sampler_free(p); }
};

// The Java callback is intentionally cheap: it only updates a volatile string.
// A UI-side timer refreshes the screen every 500ms, even during a blocked decode.
struct Progress {
    JNIEnv *env;
    jclass clazz;
    jmethodID method;
    Clock::time_point begin;
    Clock::time_point deadline;
    bool stopped_by_timeout = false;

    Progress(JNIEnv *e, jclass c)
            : env(e), clazz(c),
              method(e->GetStaticMethodID(c, "onNativeProgress",
                                          "(Ljava/lang/String;)V")),
              begin(Clock::now()), deadline(begin + std::chrono::seconds(90)) {
        if (!method) throw std::runtime_error("Missing Java progress callback");
    }

    bool expired() {
        if (Clock::now() >= deadline) {
            stopped_by_timeout = true;
            return true;
        }
        return cancelled.load(std::memory_order_relaxed);
    }

    void send(const std::string &stage) {
        jstring msg = env->NewStringUTF(stage.c_str());
        if (!msg) return;
        env->CallStaticVoidMethod(clazz, method, msg);
        env->DeleteLocalRef(msg);
    }

    void check() {
        if (expired()) {
            throw std::runtime_error(stopped_by_timeout
                ? "90秒の制限に達したので停止した" : "ユーザーが中断した");
        }
    }
};

static bool cpu_abort(void *userdata) {
    return static_cast<Progress *>(userdata)->expired();
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                         const std::string &prompt) {
    int n = llama_tokenize(vocab, prompt.c_str(),
                           static_cast<int32_t>(prompt.size()),
                           nullptr, 0, true, true);
    if (n >= 0) throw std::runtime_error("Tokenizer capacity query failed");
    std::vector<llama_token> result(static_cast<size_t>(-n));
    n = llama_tokenize(vocab, prompt.c_str(),
                       static_cast<int32_t>(prompt.size()), result.data(),
                       static_cast<int32_t>(result.size()), true, true);
    if (n <= 0) throw std::runtime_error("Tokenization failed");
    result.resize(static_cast<size_t>(n));
    return result;
}

struct Run {
    int prompt_tokens = 0;
    int output_tokens = 0;
    double prefill_ms = 0;
    double decode_ms = 0;
    double sample_ms = 0;
    std::string translation;
};

static Run run_one(llama_model *model, const std::vector<llama_token> &tokens,
                   int threads, int limit, const std::string &stage, Progress &progress) {
    progress.check();
    progress.send(stage + "：推論コンテキスト初期化");

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.no_perf = true;
    // Supported by pinned llama.cpp and called inside CPU compute.
    // This also interrupts exceptionally long single-token operations.
    params.abort_callback = cpu_abort;
    params.abort_callback_data = &progress;

    std::unique_ptr<llama_context, ContextDeleter> ctx(llama_init_from_model(model, params));
    if (!ctx) throw std::runtime_error("Context initialization failed");
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler(llama_sampler_init_greedy());
    if (!sampler) throw std::runtime_error("Greedy sampler initialization failed");
    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) throw std::runtime_error("Model vocabulary unavailable");
    if (tokens.size() > 512 || tokens.size() + static_cast<size_t>(limit) >= 2048)
        throw std::runtime_error("Input exceeds benchmark context limit");

    Run result;
    result.prompt_tokens = static_cast<int>(tokens.size());

    progress.send(stage + "：入力 " + std::to_string(tokens.size()) + " tokens を処理中");
    const auto prefill_start = Clock::now();
    llama_batch prefill = llama_batch_get_one(
        const_cast<llama_token *>(tokens.data()), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx.get(), prefill) != 0) {
        progress.check();
        throw std::runtime_error("Prefill decode failed");
    }
    result.prefill_ms = elapsed_ms(prefill_start);
    progress.check();

    const auto sample_start = Clock::now();
    llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
    result.sample_ms = elapsed_ms(sample_start);

    // Update after every token. JNI callback itself does not block on the UI.
    for (int i = 0; i < limit; ++i) {
        progress.check();
        if (llama_vocab_is_eog(vocab, token)) {
            progress.send(stage + "：EOS、" + std::to_string(result.output_tokens)
                          + " tokensで終了");
            break;
        }

        char buffer[8192];
        int len = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
        if (len > 0) result.translation.append(buffer, static_cast<size_t>(len));
        ++result.output_tokens;

        progress.send(stage + "：生成 " + std::to_string(result.output_tokens)
                      + "/" + std::to_string(limit) + " tokens");
        if (i + 1 >= limit) break;

        const auto decode_start = Clock::now();
        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), next) != 0) {
            progress.check();
            throw std::runtime_error("Incremental decode failed");
        }
        token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        result.decode_ms += elapsed_ms(decode_start);
    }
    return result;
}

static jstring execute_benchmark(JNIEnv *env, jclass clazz, jstring model_path,
                                 jstring input, jstring target_language,
                                 jint thread_count, jint max_tokens) {
    cancelled.store(false, std::memory_order_relaxed);
    try {
        Progress progress(env, clazz);
        progress.send("準備中：llama.cppバックエンド初期化");
        init_llama();

        const std::string path = utf8(env, model_path);
        const std::string text = utf8(env, input);
        const std::string target = utf8(env, target_language);
        if (path.empty() || text.empty()) throw std::runtime_error("Missing GGUF or input");
        if (target != "en" && target != "ja") throw std::runtime_error("Target must be en or ja");
        const int threads = std::max(1, std::min(8, static_cast<int>(thread_count)));
        const int output_cap = std::max(4, std::min(128, static_cast<int>(max_tokens)));

        progress.send("モデル読み込み中：GGUFをメモリへマッピング");
        const auto load_start = Clock::now();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, ModelDeleter> model(
            llama_model_load_from_file(path.c_str(), model_params));
        if (!model) throw std::runtime_error("Could not load GGUF model");
        const double load_ms = elapsed_ms(load_start);
        progress.check();

        // PLaMo 3 Translate uses PFML3 format rather than ChatML.
        // A fixed prompt format ensures that all backends see identical tokens.
        const std::string prompt =
            "<|plamo:tag|>dataset<|plamo:msg|>translation"
            "<|plamo:tag|>input<|plamo:msg|>" + text +
            "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" +
            target + "<|plamo:msg|>";

        progress.send("入力トークン化中");
        auto tokens = tokenize(llama_model_get_vocab(model.get()), prompt);
        progress.check();

        // A short 2-token warmup (rather than 64/128 generated tokens) prevents
        // repeated long translations when the goal is only to benchmark speed.
        Run warmup = run_one(model.get(), tokens, threads, 2, "ウォームアップ", progress);
        (void) warmup;
        progress.check();

        Run measured = run_one(model.get(), tokens, threads, output_cap,
                               "本計測", progress);
        const int decoding_steps = std::max(0, measured.output_tokens - 1);
        const double prefill_rate = measured.prefill_ms > 0
             ? measured.prompt_tokens * 1000.0 / measured.prefill_ms : 0.0;
        const double decode_rate = measured.decode_ms > 0
             ? decoding_steps * 1000.0 / measured.decode_ms : 0.0;

        std::ostringstream out;
        out << std::fixed << std::setprecision(2);
        out << "PLaMo Pixel unified benchmark\n";
        out << "Model load: " << load_ms << " ms\n";
        out << "Threads: " << threads << "\n";
        out << "Prompt tokens: " << measured.prompt_tokens << "\n";
        out << "Output tokens: " << measured.output_tokens << "\n";
        out << "Prefill: " << measured.prefill_ms << " ms / "
            << prefill_rate << " tok/s\n";
        out << "Decode: " << measured.decode_ms << " ms / "
            << decode_rate << " tok/s\n";
        out << "First sample: " << measured.sample_ms << " ms\n";
        out << "Output cap: " << output_cap << " tokens\n";
        out << "Translation:\n" << measured.translation << "\n";
        progress.send("完了：結果を表示中");
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception &error) {
        std::string msg = std::string("BENCHMARK STOPPED: ") + error.what();
        return env->NewStringUTF(msg.c_str());
    }
}

#if defined(PLAMO_VARIANT_GENERIC)
extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeBenchmarkGeneric(
    JNIEnv *env, jclass clazz, jstring path, jstring input, jstring target, jint threads, jint cap) {
    return execute_benchmark(env, clazz, path, input, target, threads, cap);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeCancelGeneric(JNIEnv *, jclass) {
    cancelled.store(true, std::memory_order_relaxed);
}
#elif defined(PLAMO_VARIANT_DOTPROD)
extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeBenchmarkDotprod(
    JNIEnv *env, jclass clazz, jstring path, jstring input, jstring target, jint threads, jint cap) {
    return execute_benchmark(env, clazz, path, input, target, threads, cap);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeCancelDotprod(JNIEnv *, jclass) {
    cancelled.store(true, std::memory_order_relaxed);
}
#elif defined(PLAMO_VARIANT_KLEIDI)
extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeBenchmarkKleidi(
    JNIEnv *env, jclass clazz, jstring path, jstring input, jstring target, jint threads, jint cap) {
    return execute_benchmark(env, clazz, path, input, target, threads, cap);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeCancelKleidi(JNIEnv *, jclass) {
    cancelled.store(true, std::memory_order_relaxed);
}
#else
#error "A distinct PLAMO_VARIANT must be selected for each backend library"
#endif

