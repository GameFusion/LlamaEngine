#include "LlamaRuntime.h"
#include "LlamaSession.h"

#include <sstream>

// define windows stubs
#ifdef WIN32
#define strdup _strdup
#endif

// Constructor initializes pointers to null
LlamaRuntime::LlamaRuntime() : model(nullptr) {}

// Destructor ensures proper resource cleanup
LlamaRuntime::~LlamaRuntime() {


    if (model) {
        llama_model_free(model);
        model = nullptr;
    }

    // Free allocated message content
    for (auto &msg : messages) {
        free(const_cast<char *>(msg.content));
    }
}

// Public method to load the model with default parameters
bool LlamaRuntime::loadModel() {
    return loadModelInternal(modelPath, 99, context_size);
}

// Internal method to load the model with custom parameters
bool LlamaRuntime::loadModelInternal(const std::string &modelPath, int ngl, int n_ctx) {
    context_size = n_ctx;
    error_.clear();

    logMessage("Loading Model context(" + std::to_string(n_ctx) + "): " + modelPath);

    // Set up logging callback
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

    // Get the model vocabulary
    vocab = llama_model_get_vocab(model);

    // Initialize the context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    // Check if a session already exists, create a default one if there is none
    if(sessions.empty())
    {
        sessions[0] = new LlamaSession("0", nullptr, nullptr);

    }

    // Recreate the context: For each active session, you need to create a new context using the new model.
    // After changing the model, recreate the sampler to match the new model’s parameters.
    // Recreate context and sampler for each active session
    for (auto& [sessionId, session] : sessions) {

        // clear (free) pre existing sampler and context in session
        session->clearSampler();
        session->clearContext();

        // Create new context for the session with the new model
        session->ctx = llama_new_context_with_model(model, ctx_params);
        if (!session->ctx) {
            logError("Failed to recreate context for session " + sessionId);
            error_ = "Failed to recreate context";
            return false;
        }

        // Access the maximum context size
        logMessage("Maximum context size: " + std::to_string(llama_n_ctx(session->ctx)));

        // Initialize the sampler for the new model
        session->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(session->smpl, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(session->smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(session->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // Resize formatted buffer for context size
    formatted.resize(n_ctx);

    return true;
}

// Setter for model path
void LlamaRuntime::setModelPath(const std::string &path) {
    modelPath = path;
}

// Setter for context size
void LlamaRuntime::setContextSize(int size) {
    context_size = size;
}

// Setter for temperature parameter
void LlamaRuntime::setTemperature(float temp) {
    temperature = temp;
}

// Setter for top-K sampling
void LlamaRuntime::setTopK(float k) {
    topK = k;
}

// Setter for top-P sampling
void LlamaRuntime::setTopP(float p) {
    topP = p;
}

// Setter for repetition penalty
void LlamaRuntime::setRepetitionPenalty(float penalty) {
    repetitionPenalty = penalty;
}

// Setter for log callback function
void LlamaRuntime::setLogCallback(LogCallback callback) {
    logCallback = callback;
}

// General logging function
void LlamaRuntime::logMessage(const std::string& message) {
    if (logCallback) {
        logCallback(message); // Send log to the callback
    } else {
        std::cerr << "[LlamaRuntime] " << message << std::endl; // Fallback if no callback is set
    }
}

// Log information messages
void LlamaRuntime::logInfo(const std::string& message) {
    logMessage("[INFO] " + message);
}

// Log warning messages
void LlamaRuntime::logWarning(const std::string& message) {
    logMessage("[WARNING] " + message);
}

// Log error messages
void LlamaRuntime::logError(const std::string& message) {
    logMessage("[ERROR] " + message);
}

// Log debug messages
void LlamaRuntime::logDebug(const std::string& message) {
    logMessage("[DEBUG] " + message);
}

/**
 * @brief Retrieves a session by ID.
 * @param session_id The session identifier.
 * @return Pointer to the session, or nullptr if not found.
 */
LlamaSession *LlamaRuntime::getSession(int session_id){
    // Try to find the session by session_id
    auto it = sessions.find(session_id);

    if (it == sessions.end()) {
        // The session doesn't exist
        return nullptr;
    }

    return it->second;
}

/**
 * @brief Generates a response using the given session.
 * @param session_id The session identifier.
 * @param input_prompt The input text prompt.
 * @param callback Function to handle generated response chunks.
 * @param userData Custom user data for the callback.
 * @return True if successful, false otherwise.
 */
bool LlamaRuntime::generateResponse(int session_id, const std::string &input_prompt, void (*callback)(const char*, void *userData), void *userData) {

    LlamaSession *session = getSession(session_id);
    if (session == nullptr) {
        // The session exists but is nullptr (invalid session)
        error_ = "Error: Session is invalid.";
        logError(error_);
        return false;
    }

    llama_context* ctx = session->ctx;
    llama_sampler *smpl = session->smpl;

    if (!ctx || !model || !vocab) {
        error_ = "Error: Model not loaded.";
        logError(error_);
        return false;
    }

    // Log context and KV cache usage before adding new message
    int n_ctx_total = llama_n_ctx(ctx);
    int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
    logDebug("Total context size: " + std::to_string(n_ctx_total) + "\n");
    logDebug("KV Cache used: " + std::to_string(n_ctx_used) + "\n");

    // Log current chat history size
    logDebug("Messages in history: " + std::to_string(messages.size())+ "\n");

    // add the user input to the message list and format it
    messages.push_back({"user", strdup(input_prompt.c_str())});

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

    // Log tokenized prompt
    logDebug("Tokenized prompt: " + prompt+ "\n");

    // generate a response
    if (!generate(ctx, smpl, prompt, callback, userData))
    {
        return false;
    }

    // add the response to the messages, this is the history context used to provide llm with context in future prompts
    messages.push_back({"assistant", strdup(response.c_str())});

    /*
    int prev_len = llama_chat_apply_template(llama_model_chat_template(model, nullptr), messages.data(), messages.size(), false, nullptr, 0);
    if (prev_len < 0) {
        error_ = "Error: failed to apply the chat template";
        logError(error_);
        return false;
    }
    */

    return true;
}

bool isValidUtf8(const std::string& str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.c_str());
    int num = 0;
    for (size_t i = 0; i < str.size(); i++) {
        if (num == 0) {
            if ((bytes[i] & 0x80) == 0) continue;  // ASCII (0xxxxxxx)
            else if ((bytes[i] & 0xE0) == 0xC0) num = 1; // 2-byte (110xxxxx)
            else if ((bytes[i] & 0xF0) == 0xE0) num = 2; // 3-byte (1110xxxx)
            else if ((bytes[i] & 0xF8) == 0xF0) num = 3; // 4-byte (11110xxx)
            else return false;  // Invalid first byte
        } else {
            if ((bytes[i] & 0xC0) != 0x80) return false; // Not a valid continuation byte
            num--;
        }
    }
    return num == 0;
}

// Generates a response token by token
bool LlamaRuntime::generate(llama_context* ctx, llama_sampler *smpl, const std::string &prompt, void (*callback)(const char*, void *), void *userData) {
    response.clear(); // TODO move to LlamaSession

    const bool is_first = llama_get_kv_cache_used_cells(ctx) == 0;

    std::vector<llama_token> prompt_tokens = tokenizePrompt(prompt, is_first);
    if (prompt_tokens.empty()) {
        error_ = "Error: Failed to tokenize the prompt";
        logError(error_);
        return false;
    }

    logDebug("Total tokens in prompt: " + std::to_string(prompt_tokens.size())+ "\n");

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) {
        int n_ctx_total = llama_n_ctx(ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);

        logDebug("KV Cache before decoding: " + std::to_string(n_ctx_used) + " / " + std::to_string(n_ctx_total)+ "\n");

        if (n_ctx_used + batch.n_tokens > n_ctx_total) {
            logError("Context size exceeded! Used: " + std::to_string(n_ctx_used) + ", Limit: " + std::to_string(n_ctx_total)+ "\n");
            return false;
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

        char buf[256] = {0};  // Ensures all bytes are initialized to '\0'

        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            error_ = "Error: failed to convert token to piece";
            logError(error_);
            return false;
        }
        //std::string piece(buf, n);
        std::string piece;
        piece.assign(buf, n);  // More robust than std::string(buf, n)

        if (!isValidUtf8(piece)) {  // Validate UTF-8 (function provided below)
            logDebug("Warning: Token ID " + std::to_string(new_token_id) + " produced invalid UTF-8, skipping.");
            continue;
        }
        else
        {
            if(n > 0)
                logDebug("Sampled Token ID: " + std::to_string(new_token_id) + " n(" + std::to_string(n)  + ") -> \"" + piece + "\"\n");
            else
                logDebug("Sampled Token ID with N=0: " + std::to_string(new_token_id) + "\n");
            logDebug("KV Cache after decoding: " + std::to_string(llama_get_kv_cache_used_cells(ctx)) + " / " + std::to_string(n_ctx_total)+ "\n");

            if (callback)
                callback(piece.c_str(), userData);
        }

        response += piece;
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    return true;
}

const std::string LlamaRuntime::getResponse() {
    return response;
}

std::vector<llama_token> LlamaRuntime::tokenizePrompt(const std::string &prompt, bool is_first) {
    // Determine the number of tokens required for the given prompt
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);

    // If tokenization fails, return an empty vector
    if (n_prompt_tokens < 0) {
        return {};
    }

    // Allocate a vector to store the tokens
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    // Perform actual tokenization and store the result in the vector
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        return {}; // Return an empty vector on failure
    }

    return prompt_tokens;
}

GGUFMetadata LlamaRuntime::parseGGUF(const std::string& filepath, void (*messageCallback)(const char* message)) {
    GGUFMetadata metadata;
    struct gguf_init_params params = { true };

    // Initialize GGUF context from file
    struct gguf_context *ctx = gguf_init_from_file(filepath.c_str(), params);
    if (!ctx) {

        std::string error = "[ERROR]: Failed to load GGUF file: " + filepath+"\n";
        if (messageCallback)
            messageCallback(error.c_str());
        return metadata; // Return empty metadata on failure
    }

    // Retrieve the number of metadata keys
    uint32_t key_count = gguf_get_n_kv(ctx);
    //logMessage("GGUF Metadata Keys: "+std::to_string(key_count)+"\n");
    std::string message = "GGUF Metadata Keys: " + std::to_string(key_count) + "\n";
    if (messageCallback)
        messageCallback(message.c_str());

    // Iterate through each metadata key and store relevant values
    for (uint32_t i = 0; i < key_count; ++i) {
        const char *key = gguf_get_key(ctx, i);
        int64_t key_id = gguf_find_key(ctx, key);

        // Ensure the key exists
        if (key_id < 0) {
            std::string error = "[ERROR]: Failed to find key: "+std::to_string(key_id)+"\n";
            if (messageCallback)
                messageCallback(error.c_str());
            continue;
        }

        // Determine the value type for the key
        enum gguf_type type = gguf_get_kv_type(ctx, key_id);
        GGUFMetadataEntry entry;

        if (type == GGUF_TYPE_UINT32) {
            // Handle UINT32 type values
            uint32_t value = gguf_get_val_u32(ctx, key_id);
            entry.type = GGUFType::TYPE_UINT32;
            entry.ivalue = value;
        }
        else if (type == GGUF_TYPE_STRING) {
            // Handle STRING type values
            const char *value = gguf_get_val_str(ctx, key_id);
            entry.type = GGUFType::TYPE_STRING;
            entry.svalue = value;
        }
        else {
            // Handle unknown types gracefully
            std::string message = "Unknown type for key: " + std::to_string(key_id) + "\n";
            if (messageCallback)
                messageCallback(message.c_str());
            
            entry.type = GGUFType::TYPE_UNKNOWN;
        }

        // Store the metadata entry
        metadata.entries[key] = entry;
    }

    // Clean up GGUF context
    gguf_free(ctx);

    return metadata;
}

std::string LlamaRuntime::getContextInfo() {
    std::stringstream ss;
    ss << "Llama Context Information\n";
    ss << "--------------------------\n";
    ss << "Model Path: " << modelPath << "\n";
    ss << "Total Context Size: " << context_size << " tokens\n";

    LlamaSession *session = getSession(0);

    int total_char_count = 0;
    int total_token_count = 0;

    for (auto& [sessionId, session] : sessions) {

        ss << "Total Messages: " << messages.size() << "\n";
        for (size_t i = 0; i < messages.size(); i++) {
            size_t char_count = strlen(messages[i].content);
            int token_count = llama_tokenize(vocab, messages[i].content, char_count, nullptr, 0, true, false);
            total_char_count += char_count;
            total_token_count += token_count;

            ss << "Message " << i << " | Role: " << messages[i].role
               << " | Size: " << char_count << " chars, " << token_count << " tokens\n";
        }

        int used_size = total_token_count;
        int remaining_size = context_size - used_size;

        ss << "\nUsed Context Size: " << used_size << " tokens\n";
        ss << "Remaining Context Size: " << remaining_size << " tokens\n\n";
    }

    ss << "Details per Prompt Response:\n";
    for (auto& [sessionId, session] : sessions) {

        ss << "Session ID: " << session->sessionId << "\n";
        for (const auto& msg : session->history) {
            ss << "  - Prompt: " << msg.prompt << "\n";
            ss << "    Response: " << msg.response << "\n";
            //ss << "    Token Count: " << msg.tokenCount << "\n";
            ss << "    Timestamp: " << msg.timestamp << "\n";
        }
    }

    return ss.str();
}
