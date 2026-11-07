// Experimental CPU-prefill and Vulkan-decode pipeline.
// Included by bench.cpp only in the PLAMO_VARIANT_HYBRID build.
static std::unique_ptr<llama_model, ModelDeleter> hybrid_cpu_model;

struct HybridRun {
    Run run;
    double gpu_context_ms = 0;
    double transfer_ms = 0;
    size_t transfer_bytes = 0;
};

static HybridRun hybrid_one(llama_model *cpu_model, llama_model *gpu_model,
                            const std::vector<llama_token> &tokens, int threads,
                            int limit, const std::string &stage, Progress &progress) {
    progress.check();
    const int needed = static_cast<int>(tokens.size()) + limit + kContextMargin;
    if (needed > kContextLimit) throw std::runtime_error("Hybrid token budget over context");

    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::min(kContextLimit, std::max(512, ((needed + 255) / 256) * 256));
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.no_perf = true;
    params.abort_callback = cpu_abort;
    params.abort_callback_data = &progress;

    HybridRun measured;
    progress.send(stage + "：CPU・GPUコンテキスト初期化");
    const auto init_start = Clock::now();
    std::unique_ptr<llama_context, ContextDeleter> gpu_ctx(
        llama_init_from_model(gpu_model, params));
    if (!gpu_ctx) throw std::runtime_error("GPU context init failed");
    measured.gpu_context_ms = elapsed_ms(init_start);
    std::unique_ptr<llama_context, ContextDeleter> cpu_ctx(
        llama_init_from_model(cpu_model, params));
    if (!cpu_ctx) throw std::runtime_error("CPU context init failed");
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler(llama_sampler_init_greedy());
    if (!sampler) throw std::runtime_error("Greedy sampler init failed");
    const llama_vocab *vocab = llama_model_get_vocab(cpu_model);
    if (!vocab) throw std::runtime_error("Model vocabulary unavailable");

    Run &result = measured.run;
    result.prompt_tokens = static_cast<int>(tokens.size());
    progress.send(stage + "：KleidiAI Prefill");
    const auto prefill_start = Clock::now();
    for (size_t pos = 0; pos < tokens.size(); pos += 512) {
        progress.check();
        const int count = static_cast<int>(std::min<size_t>(512, tokens.size() - pos));
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens.data() + pos), count);
        if (llama_decode(cpu_ctx.get(), batch) != 0) {
            progress.check();
            throw std::runtime_error("CPU prefill failed");
        }
    }
    result.prefill_ms = elapsed_ms(prefill_start);
    progress.check();

    // The first output token comes from CPU logits; transfer only the state needed
    // to continue token-by-token decoding on the Vulkan backend.
    const auto sample_start = Clock::now();
    llama_token token = llama_sampler_sample(sampler.get(), cpu_ctx.get(), -1);
    result.sample_ms = elapsed_ms(sample_start);

    progress.send(stage + "：KVキャッシュをCPUからGPUへ転送");
    const auto transfer_start = Clock::now();
    const size_t capacity = llama_state_get_size(cpu_ctx.get());
    if (capacity == 0 || capacity > 1024ULL * 1024ULL * 1024ULL)
        throw std::runtime_error("Invalid KV state size");
    std::vector<uint8_t> state(capacity);
    const size_t bytes = llama_state_get_data(cpu_ctx.get(), state.data(), state.size());
    if (bytes == 0 || bytes > state.size())
        throw std::runtime_error("KV state serialization failed");
    progress.check();
    const size_t loaded = llama_state_set_data(gpu_ctx.get(), state.data(), bytes);
    if (loaded != bytes)
        throw std::runtime_error("KV state cannot be restored in GPU context");
    measured.transfer_bytes = bytes;
    measured.transfer_ms = elapsed_ms(transfer_start);
    cpu_ctx.reset();
    state.clear();
    state.shrink_to_fit();
    progress.check();

    progress.send(stage + "：Vulkan Decode");
    for (int i = 0; i < limit; ++i) {
        progress.check();
        if (llama_vocab_is_eog(vocab, token)) break;
        char buffer[8192];
        const int len = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
        if (len > 0) result.translation.append(buffer, static_cast<size_t>(len));
        ++result.output_tokens;
        for (const char *end : {"<|plamo:tag|>", "<|plamo:bos|>"}) {
            const size_t at = result.translation.find(end);
            if (at != std::string::npos) {
                result.translation.resize(at);
                return measured;
            }
        }
        if (result.output_tokens == 1 || result.output_tokens % 16 == 0) {
            std::ostringstream message;
            message << stage << "：Vulkan生成 " << result.output_tokens << "/" << limit;
            if (result.decode_ms > 0) {
                message << " / " << std::fixed << std::setprecision(1)
                        << result.decode_steps * 1000.0 / result.decode_ms << " tok/s";
            }
            progress.send(message.str());
        }
        if (i + 1 >= limit) {
            result.reached_cap = true;
            break;
        }
        const auto decode_start = Clock::now();
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(gpu_ctx.get(), batch) != 0) {
            progress.check();
            throw std::runtime_error("Hybrid GPU decode failed");
        }
        token = llama_sampler_sample(sampler.get(), gpu_ctx.get(), -1);
        result.decode_ms += elapsed_ms(decode_start);
        ++result.decode_steps;
    }
    return measured;
}

static jstring hybrid_execute(JNIEnv *env, jclass clazz, jstring path_js, jstring input_js,
                              jstring language_js, jint threads_js, jint cap_js) {
    std::lock_guard<std::mutex> lock(model_mutex);
    cancelled.store(false, std::memory_order_relaxed);
    try {
        Progress progress(env, clazz);
        init_llama();
        const std::string path = utf8(env, path_js);
        const std::string input = utf8(env, input_js);
        const std::string target = utf8(env, language_js);
        if (path.empty() || input.empty()) throw std::runtime_error("Missing model or input");
        if (target != "ja" && target != "en") throw std::runtime_error("Invalid target language");
        const int threads = std::max(1, std::min(8, static_cast<int>(threads_js)));
        const int cap = std::max(16, std::min(2048, static_cast<int>(cap_js)));
        const bool reused = hybrid_cpu_model && cached_model && cached_model_path == path;
        double model_load_ms = 0;
        if (!reused) {
            progress.send("KleidiAI CPUモデルとVulkan GPUモデルを読み込み中");
            hybrid_cpu_model.reset();
            cached_model.reset();
            cached_model_path.clear();
            const auto load_start = Clock::now();
            llama_model_params cpu_params = llama_model_default_params();
            cpu_params.n_gpu_layers = 0;
            hybrid_cpu_model.reset(llama_model_load_from_file(path.c_str(), cpu_params));
            if (!hybrid_cpu_model) throw std::runtime_error("Hybrid CPU model load failed");
            progress.check();
            llama_model_params gpu_params = llama_model_default_params();
            gpu_params.n_gpu_layers = 99;
            cached_model.reset(llama_model_load_from_file(path.c_str(), gpu_params));
            if (!cached_model) throw std::runtime_error("Hybrid Vulkan model load failed");
            cached_model_path = path;
            model_load_ms = elapsed_ms(load_start);
        }
        progress.check();
        const llama_vocab *vocab = llama_model_get_vocab(hybrid_cpu_model.get());
        if (!vocab) throw std::runtime_error("Hybrid vocab unavailable");
        const auto segments = split_only_if_needed(vocab, input, target, cap, progress);

        Run total;
        double transfer_ms = 0;
        double context_ms = 0;
        size_t transfer_bytes = 0;
        std::ostringstream translated;
        for (size_t index = 0; index < segments.size(); ++index) {
            const std::string stage = "Hybrid " + std::to_string(index + 1) +
                                      "/" + std::to_string(segments.size());
            const auto prompt = tokenize(vocab, make_prompt(segments[index], target));
            HybridRun item = hybrid_one(hybrid_cpu_model.get(), cached_model.get(),
                                        prompt, threads, cap, stage, progress);
            total.prompt_tokens += item.run.prompt_tokens;
            total.output_tokens += item.run.output_tokens;
            total.decode_steps += item.run.decode_steps;
            total.prefill_ms += item.run.prefill_ms;
            total.decode_ms += item.run.decode_ms;
            total.sample_ms += item.run.sample_ms;
            total.reached_cap = total.reached_cap || item.run.reached_cap;
            transfer_ms += item.transfer_ms;
            context_ms += item.gpu_context_ms;
            transfer_bytes += item.transfer_bytes;
            if (index > 0) translated << "\n";
            translated << item.run.translation;
        }
        std::ostringstream out;
        out << std::fixed << std::setprecision(2);
        out << "PLaMo Pixel Hybrid benchmark v0.5.0\n";
        out << "Pipeline: KleidiAI CPU Prefill -> Vulkan GPU Decode\n";
        out << "Model load: " << model_load_ms << " ms\n";
        out << "Model cache: " << (reused ? "reused" : "cold") << "\n";
        out << "Threads: " << threads << "\n";
        out << "Context limit: " << kContextLimit << " tokens\n";
        out << "Segments: " << segments.size() << " (only split on overflow)\n";
        out << "Prompt tokens: " << total.prompt_tokens << "\n";
        out << "Output tokens: " << total.output_tokens << "\n";
        out << "CPU Prefill: " << total.prefill_ms << " ms / "
            << (total.prefill_ms > 0 ? total.prompt_tokens * 1000.0 / total.prefill_ms : 0) << " tok/s\n";
        out << "CPU->GPU KV transfer: " << transfer_ms << " ms / "
            << transfer_bytes << " bytes\n";
        out << "GPU context init: " << context_ms << " ms\n";
        out << "GPU Decode: " << total.decode_ms << " ms / "
            << (total.decode_ms > 0 ? total.decode_steps * 1000.0 / total.decode_ms : 0) << " tok/s\n";
        out << "First sample (CPU): " << total.sample_ms << " ms\n";
        out << "Output cap: " << cap << " tokens per segment\n";
        out << "Stopped by output cap: " << (total.reached_cap ? "YES" : "NO") << "\n";
        out << "Translation:\n" << translated.str() << "\n";
        progress.send("ハイブリッド計測完了");
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception &e) {
        return env->NewStringUTF((std::string("HYBRID STOPPED: ") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeBenchmarkHybrid(
    JNIEnv *env, jclass cls, jstring path, jstring input, jstring language, jint threads, jint cap) {
    return hybrid_execute(env, cls, path, input, language, threads, cap);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeCancelHybrid(JNIEnv *, jclass) {
    cancelled.store(true, std::memory_order_relaxed);
}
extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_plamobench_MainActivity_nativeReleaseHybrid(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(model_mutex);
    hybrid_cpu_model.reset();
    cached_model.reset();
    cached_model_path.clear();
}
