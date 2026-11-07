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
              begin(Clock::now()), deadline(begin + std::chrono::seconds(300)) {
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
                ? "300秒の制限に達したので停止した" : "ユーザーが中断した");
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

// Pixel 7 ProのRAMを節約するため、どの時点でも1種類のCPUモデルだけを維持する。
// Java側はエンジン切替時に他方式のキャッシュを解放する。
static std::mutex model_mutex;
static std::unique_ptr<llama_model, ModelDeleter> cached_model;
static std::string cached_model_path;

// モデルは32Kトークンまで学習済みだが、このCPUベンチではメモリ消費を抑える。
// 入力＋出力上限が収まるなら、段落が複数あっても分割しない。
static constexpr int kContextLimit = 4096;
static constexpr int kContextMargin = 32;

static std::string make_prompt(const std::string &input, const std::string &target) {
    // PFN公式が示すPFML3翻訳プロンプト。入力言語はモデルに自動判定させる。
    return "<|plamo:tag|>dataset<|plamo:msg|>translation"
           "<|plamo:tag|>input<|plamo:msg|>" + input +
           "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" +
           target + "<|plamo:msg|>";
}

static size_t utf8_boundary(const std::string &text, size_t index) {
    index = std::min(index, text.size());
    // 日本語のUTF-8継続バイトの途中で分割しない。
    while (index > 0 && index < text.size()
           && (static_cast<unsigned char>(text[index]) & 0xc0) == 0x80) {
        --index;
    }
    return index;
}

static std::vector<std::string> split_only_if_needed(
        const llama_vocab *vocab, const std::string &input,
        const std::string &target, int output_cap, Progress &progress) {
    const size_t input_budget = static_cast<size_t>(
        kContextLimit - output_cap - kContextMargin);
    if (input_budget < 128) throw std::runtime_error("Output cap leaves insufficient context");
    auto fits = [&](const std::string &candidate) {
        return tokenize(vocab, make_prompt(candidate, target)).size() <= input_budget;
    };

    // まず全文で収まるかを判定。大多数の短文・通常の長文はここで終了。
    if (fits(input)) return {input};

    progress.send("コンテキスト上限を超過：文・段落の境界で分割中");
    std::vector<std::string> pieces;
    std::string rest = input;
    while (!rest.empty()) {
        progress.check();
        if (fits(rest)) {
            pieces.push_back(rest);
            break;
        }

        // トークン数による二分探索。文字数で恣意的に段落分割しない。
        size_t low = 1, high = rest.size(), best = 0;
        while (low <= high) {
            const size_t middle = low + (high - low) / 2;
            const size_t boundary = utf8_boundary(rest, middle);
            if (boundary > 0 && fits(rest.substr(0, boundary))) {
                best = boundary;
                low = middle + 1;
            } else {
                if (middle == 0) break;
                high = middle - 1;
            }
        }
        if (best == 0) throw std::runtime_error("Could not split input within context");

        size_t cut = best;
        const size_t earliest = (best * 2) / 3;
        // 余裕があるときは段落→改行→句点→空白の順で自然な境界を探す。
        const auto try_delimiter = [&](const std::string &delimiter) {
            const size_t found = rest.rfind(delimiter, best - 1);
            if (found == std::string::npos) return false;
            const size_t after = found + delimiter.size();
            if (after < earliest || after > best) return false;
            cut = after;
            return true;
        };
        if (!try_delimiter("\n\n") && !try_delimiter("\n")
                && !try_delimiter("。") && !try_delimiter(". ")
                && !try_delimiter(" ")) {
            cut = best;
        }
        pieces.push_back(rest.substr(0, cut));
        rest.erase(0, cut);
        if (pieces.size() >= 256)
            throw std::runtime_error("Input exceeds maximum segment count");
    }
    return pieces;
}

struct Run {
    int prompt_tokens = 0;
    int output_tokens = 0;
    int decode_steps = 0;
    bool reached_cap = false;
    double prefill_ms = 0;
    double decode_ms = 0;
    double sample_ms = 0;
    std::string translation;
};

static Run run_one(llama_model *model, const std::vector<llama_token> &tokens,
                   int threads, int limit, const std::string &stage, Progress &progress) {
    progress.check();
    if (tokens.size() + static_cast<size_t>(limit) + kContextMargin > kContextLimit)
        throw std::runtime_error("Token budget exceeds maximum context");
    progress.send(stage + "：推論コンテキスト初期化");

    llama_context_params params = llama_context_default_params();
    // 入力が短い場合に巨大なKVキャッシュを割り当てない。
    const int wanted = static_cast<int>(tokens.size()) + limit + kContextMargin;
    params.n_ctx = std::min(kContextLimit, std::max(512, ((wanted + 255) / 256) * 256));
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.no_perf = true;
    params.abort_callback = cpu_abort;
    params.abort_callback_data = &progress;

    std::unique_ptr<llama_context, ContextDeleter> ctx(llama_init_from_model(model, params));
    if (!ctx) throw std::runtime_error("Context initialization failed");
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler(llama_sampler_init_greedy());
    if (!sampler) throw std::runtime_error("Greedy sampler initialization failed");
    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) throw std::runtime_error("Model vocabulary unavailable");

    Run result;
    result.prompt_tokens = static_cast<int>(tokens.size());
    progress.send(stage + "：入力 " + std::to_string(tokens.size()) + " tokens を処理中");

    // 512トークンを超える入力も落とさず、実際のコンテキスト上限まで処理する。
    const auto prefill_start = Clock::now();
    for (size_t pos = 0; pos < tokens.size(); pos += 512) {
        progress.check();
        const int count = static_cast<int>(std::min<size_t>(512, tokens.size() - pos));
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens.data() + pos), count);
        if (llama_decode(ctx.get(), batch) != 0) {
            progress.check();
            throw std::runtime_error("Prefill decode failed");
        }
    }
    result.prefill_ms = elapsed_ms(prefill_start);
    progress.check();

    const auto sample_start = Clock::now();
    llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
    result.sample_ms = elapsed_ms(sample_start);

    for (int i = 0; i < limit; ++i) {
        progress.check();
        if (llama_vocab_is_eog(vocab, token)) break;

        char buffer[8192];
        const int len = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
        if (len > 0) result.translation.append(buffer, static_cast<size_t>(len));
        ++result.output_tokens;

        // PLaMo特有の次メッセージ開始タグが出た場合は訳文のみで止める。
        for (const char *stop : {"<|plamo:tag|>", "<|plamo:bos|>"}) {
            const size_t at = result.translation.find(stop);
            if (at != std::string::npos) {
                result.translation.resize(at);
                progress.send(stage + "：翻訳終端タグを検出");
                return result;
            }
        }

        if (result.output_tokens == 1 || result.output_tokens % 16 == 0) {
            std::ostringstream message;
            message << stage << "：生成 " << result.output_tokens << "/" << limit << " tokens";
            if (result.decode_ms > 0) {
                message << " / " << std::fixed << std::setprecision(1)
                        << (1000.0 * result.decode_steps / result.decode_ms) << " tok/s";
            }
            progress.send(message.str());
        }
        if (i + 1 >= limit) {
            result.reached_cap = true;
            break;
        }

        const auto decode_start = Clock::now();
        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), next) != 0) {
            progress.check();
            throw std::runtime_error("Incremental decode failed");
        }
        token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        result.decode_ms += elapsed_ms(decode_start);
        ++result.decode_steps;
    }
    return result;
}

static jstring execute_benchmark(JNIEnv *env, jclass clazz, jstring model_path,
                                 jstring input, jstring target_language,
                                 jint thread_count, jint max_tokens) {
    // キャッシュと推論の排他制御。キャンセル用atomicは別スレッドから更新できる。
    std::lock_guard<std::mutex> lock(model_mutex);
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
        const int output_cap = std::max(16, std::min(2048, static_cast<int>(max_tokens)));

        const bool cache_hit = cached_model && cached_model_path == path;
        double load_ms = 0;
        if (!cache_hit) {
            cached_model.reset();
            cached_model_path.clear();
            progress.send("モデル読み込み中：GGUFをメモリへマッピング");
            const auto load_start = Clock::now();
            llama_model_params model_params = llama_model_default_params();
            // Vulkan版のみGPUへ全レイヤーのオフロードを要求する。
#if defined(PLAMO_VARIANT_VULKAN)
            model_params.n_gpu_layers = 99;
#else
            model_params.n_gpu_layers = 0;
#endif
            cached_model.reset(llama_model_load_from_file(path.c_str(), model_params));
            if (!cached_model) throw std::runtime_error("Could not load GGUF model");
            cached_model_path = path;
            load_ms = elapsed_ms(load_start);
        } else {
            progress.send("モデルキャッシュ再利用：読み込みをスキップ");
        }
        progress.check();

        llama_model *model = cached_model.get();
        const llama_vocab *vocab = llama_model_get_vocab(model);
        if (!vocab) throw std::runtime_error("Model vocabulary unavailable");

        progress.send("入力トークン化・コンテキスト長を確認中");
        const auto pieces = split_only_if_needed(vocab, text, target, output_cap, progress);
        progress.check();

        // ウォームアップ用の重複翻訳は実行せず、実使用時の速度を測る。
        Run total;
        std::ostringstream translations;
        for (size_t index = 0; index < pieces.size(); ++index) {
            const std::string stage = "本計測 " + std::to_string(index + 1)
                                    + "/" + std::to_string(pieces.size());
            const auto tokens = tokenize(vocab, make_prompt(pieces[index], target));
            Run measured = run_one(model, tokens, threads, output_cap, stage, progress);
            total.prompt_tokens += measured.prompt_tokens;
            total.output_tokens += measured.output_tokens;
            total.decode_steps += measured.decode_steps;
            total.prefill_ms += measured.prefill_ms;
            total.decode_ms += measured.decode_ms;
            total.sample_ms += measured.sample_ms;
            total.reached_cap = total.reached_cap || measured.reached_cap;
            if (index > 0) translations << "\n";
            translations << measured.translation;
            progress.check();
        }

        const double prefill_rate = total.prefill_ms > 0
            ? total.prompt_tokens * 1000.0 / total.prefill_ms : 0.0;
        const double decode_rate = total.decode_ms > 0
            ? total.decode_steps * 1000.0 / total.decode_ms : 0.0;

        std::ostringstream out;
        out << std::fixed << std::setprecision(2);
        out << "PLaMo Pixel unified benchmark v0.4.1\n";
#if defined(PLAMO_VARIANT_VULKAN)
        out << "GPU offload: requested (verify on Pixel device)\n";
#else
        out << "GPU offload: disabled\n";
#endif
        out << "Model load: " << load_ms << " ms\n";
        out << "Model cache: " << (cache_hit ? "reused" : "cold") << "\n";
        out << "Threads: " << threads << "\n";
        out << "Context limit: " << kContextLimit << " tokens\n";
        out << "Segments: " << pieces.size() << " (only split on overflow)\n";
        out << "Prompt tokens: " << total.prompt_tokens << "\n";
        out << "Output tokens: " << total.output_tokens << "\n";
        out << "Prefill: " << total.prefill_ms << " ms / "
            << prefill_rate << " tok/s\n";
        out << "Decode: " << total.decode_ms << " ms / "
            << decode_rate << " tok/s\n";
        out << "First sample total: " << total.sample_ms << " ms\n";
        out << "Output cap: " << output_cap << " tokens per segment\n";
        out << "Stopped by output cap: " << (total.reached_cap ? "YES" : "NO") << "\n";
        out << "Translation:\n" << translations.str() << "\n";
        progress.send("完了：結果を表示中");
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception &error) {
        const std::string msg = std::string("BENCHMARK STOPPED: ") + error.what();
        return env->NewStringUTF(msg.c_str());
    }
}

static void release_cached_model() {
    std::lock_guard<std::mutex> lock(model_mutex);
    cached_model.reset();
    cached_model_path.clear();
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
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeReleaseGeneric(JNIEnv *, jclass) {
    release_cached_model();
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
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeReleaseDotprod(JNIEnv *, jclass) {
    release_cached_model();
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
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeReleaseKleidi(JNIEnv *, jclass) {
    release_cached_model();
}
#elif defined(PLAMO_VARIANT_VULKAN)
extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeBenchmarkVulkan(
    JNIEnv *env, jclass clazz, jstring path, jstring input, jstring target, jint threads, jint cap) {
    return execute_benchmark(env, clazz, path, input, target, threads, cap);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeCancelVulkan(JNIEnv *, jclass) {
    cancelled.store(true, std::memory_order_relaxed);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeReleaseVulkan(JNIEnv *, jclass) {
    release_cached_model();
}
#elif defined(PLAMO_VARIANT_HYBRID)
#include "hybrid_impl.inc"
#else
#error "A distinct PLAMO_VARIANT must be selected for each backend library"
#endif

