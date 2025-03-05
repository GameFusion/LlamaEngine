#include "LlamaRuntime.h"

LlamaRuntime::LlamaRuntime() : model(nullptr), smpl(nullptr), ctx(nullptr) {}

LlamaRuntime::~LlamaRuntime() {
    if (smpl) {
        llama_sampler_free(smpl);
        smpl = nullptr;
    }

    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }

    if (model) {
        llama_model_free(model);
        model = nullptr;
    }

    // Free allocated message content
    for (auto &msg : messages) {
        free(const_cast<char *>(msg.content));
    }
}

bool LlamaRuntime::loadModel() {
    return loadModelInternal(modelPath, 99, context_size);
}

bool LlamaRuntime::loadModelInternal(const std::string &modelPath, int ngl, int n_ctx) {
    context_size = n_ctx;
    error_.clear();

    logMessage("Loading Model context(" + std::to_string(n_ctx) + "): " + modelPath);

    llama_log_set([](enum ggml_log_level level, const char *text, void *thisContext) {

        ((LlamaRuntime*)thisContext)->logMessage(text);

        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, this);

    // Load dynamic backends
    ggml_backend_load_all();

    // Initialize the model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    // Load the model
    model = llama_load_model_from_file(modelPath.c_str(), model_params);
    if (!model) {
        logError("Failed to load model");
        error_ = "Failed to load model file";
        return false;
    }

    // Get the vocabulary
    vocab = llama_model_get_vocab(model);

    // Initialize the context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    // Create the context with the model
    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        logError("Failed to create context");
        error_ = "Failed to create context";
        return false;
    }

    // Access the maximum context size
    logMessage("Maximum context size: " + std::to_string(llama_n_ctx(ctx)));

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    formatted.resize(n_ctx);

    return true;
}

void LlamaRuntime::setModelPath(const std::string &path) {
    modelPath = path;
}

void LlamaRuntime::setContextSize(int size) {
    context_size = size;
}

void LlamaRuntime::setTemperature(float temp) {
    temperature = temp;
}

void LlamaRuntime::setTopK(float k) {
    topK = k;
}

void LlamaRuntime::setTopP(float p) {
    topP = p;
}

void LlamaRuntime::setRepetitionPenalty(float penalty) {
    repetitionPenalty = penalty;
}

void LlamaRuntime::setLogCallback(LogCallback callback) {
    logCallback = callback;
}

void LlamaRuntime::logMessage(const std::string& message) {
    if (logCallback) {
        logCallback(message); // Send log to the callback
    } else {
        std::cerr << "[LlamaRuntime] " << message << std::endl; // Fallback if no callback is set
    }
}

void LlamaRuntime::logInfo(const std::string& message) {
    logMessage("[INFO] " + message);
}

void LlamaRuntime::logWarning(const std::string& message) {
    logMessage("[WARNING] " + message);
}

void LlamaRuntime::logError(const std::string& message) {
    logMessage("[ERROR] " + message);
}

bool LlamaRuntime::generateResponse(const std::string &input_prompt, void (*callback)(const char*, void *userData), void *userData) {
    if (!ctx || !model || !vocab) {
        error_ = "Error: Model not loaded.";
        logError(error_);
        return false;
    }

    const std::string user_input = input_prompt;
    // add the user input to the message list and format it
    messages.push_back({"user", strdup(user_input.c_str())});

    int new_len = llama_chat_apply_template(llama_model_chat_template(model, nullptr), messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(llama_model_chat_template(model, nullptr), messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        error_ = "Error: failed to apply the chat template";
        logError(error_);
        return false;
    }

    // remove previous messages to obtain the prompt to generate the response
    std::string prompt(formatted.begin(), formatted.begin() + new_len);

    // generate a response
    if (!generate(prompt, callback, userData))
    {
        return false;
    }

    // add the response to the messages
    messages.push_back({"assistant", strdup(response.c_str())});
    int prev_len = llama_chat_apply_template(llama_model_chat_template(model, nullptr), messages.data(), messages.size(), false, nullptr, 0);
    if (prev_len < 0) {
        error_ = "Error: failed to apply the chat template";
        logError(error_);
        return false;
    }

    //return response;
    return false;
}

bool LlamaRuntime::generate(const std::string &prompt, void (*callback)(const char*, void *), void *userData) {
    //std::string response;

    response.clear();

    const bool is_first = llama_get_kv_cache_used_cells(ctx) == 0;

    std::vector<llama_token> prompt_tokens = tokenizePrompt(prompt, is_first);
    if (prompt_tokens.empty()) {
        error_ = "Error: Failed to tokenize the prompt";
        logError(error_);
        return false;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) {
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            if (response.length()) {
                error_ = "Error: Failed to tokenize the prompt";
                logError(error_);
                return false;
            } else {
                error_ = "Error: context size exceeded";
                logError(error_);
                return false;
            }
        }

        if (llama_decode(ctx, batch)) {
            error_ = "Error: failed to decode";
            logError(error_);
            return false;
        }

        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            error_ = "Error: failed to convert token to piece";
            logError(error_);
            return false;
        }
        std::string piece(buf, n);
        if (callback) callback(piece.c_str(), userData);

        response += piece;
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    // return response;
    return true;
}

std::vector<llama_token> LlamaRuntime::tokenizePrompt(const std::string &prompt, bool is_first) {
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
    if (n_prompt_tokens < 0) {
        return {};
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        return {};
    }

    return prompt_tokens;
}

GGUFMetadata LlamaRuntime::parseGGUF(const std::string& filepath, void (*messageCallback)(const char* message)) {
    GGUFMetadata metadata;
    struct gguf_init_params params = { true };

    struct gguf_context *ctx = gguf_init_from_file(filepath.c_str(), params);
    if (!ctx) {

        std::string error = "[ERROR]: Failed to load GGUF file: " + filepath+"\n";
        if (messageCallback)
            messageCallback(error.c_str());
        return metadata;
    }

    uint32_t key_count = gguf_get_n_kv(ctx);
    //logMessage("GGUF Metadata Keys: "+std::to_string(key_count)+"\n");
    std::string message = "GGUF Metadata Keys: " + std::to_string(key_count) + "\n";
    if (messageCallback)
        messageCallback(message.c_str());

    for (uint32_t i = 0; i < key_count; ++i) {
        const char *key = gguf_get_key(ctx, i);
        int64_t key_id = gguf_find_key(ctx, key);

        if (key_id < 0) {
            std::string error = "[ERROR]: Failed to find key: "+std::to_string(key_id)+"\n";
            if (messageCallback)
                messageCallback(error.c_str());
            continue;
        }

        enum gguf_type type = gguf_get_kv_type(ctx, key_id);
        GGUFMetadataEntry entry;

        if (type == GGUF_TYPE_UINT32) {
            uint32_t value = gguf_get_val_u32(ctx, key_id);
            entry.type = GGUFType::TYPE_UINT32;
            entry.ivalue = value;
        }
        else if (type == GGUF_TYPE_STRING) {
            const char *value = gguf_get_val_str(ctx, key_id);
            entry.type = GGUFType::TYPE_STRING;
            entry.svalue = value;
        }
        else {
            std::string message = "Unknown type for key: " + std::to_string(key_id) + "\n";
            if (messageCallback)
                messageCallback(message.c_str());
            
            entry.type = GGUFType::TYPE_UNKNOWN;
        }

        metadata.entries[key] = entry;
    }

    gguf_free(ctx);

    return metadata;
}
