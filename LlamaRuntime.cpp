#include "LlamaRuntime.h"
#include "LlamaSession.h"

#include "common.h"

#include "LlamaRuntimeVision.h"

#include <sstream>

// define windows stubs
#ifdef WIN32
#define strdup _strdup
#endif

// Constructor initializes pointers to null
LlamaRuntime::LlamaRuntime() : model(nullptr), clip_model(nullptr) {}

// Destructor ensures proper resource cleanup
LlamaRuntime::~LlamaRuntime() {


    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
/*
    // Free allocated message content
    for (auto &msg : messages) {
        free(const_cast<char *>(msg.content));
    }
*/
    if (clip_model) {
        clip_free(clip_model);
        clip_model = nullptr;
    }

    for (auto& [sessionId, session] : sessions) {
        delete session;
    }
    sessions.clear();
}

// Check if vision model is loaded
bool LlamaRuntime::isVisionModelLoaded() const {
    return clip_model != nullptr;
}

// Load CLIP model for image processing
bool LlamaRuntime::loadClipModel(const std::string &clip_model_path, void (*callback)(const char*, void *userData), void *userData) {
    clipModelPath = clip_model_path;

    if (clip_model) {
        clip_free(clip_model);
        clip_model = nullptr;
    }

    logInfo("Loading CLIP model: " + clipModelPath);
    try {
        // Attempt to load the CLIP model
        clip_model = clip_model_load(clipModelPath.c_str(), false);

        // Check if loading was successful
        if (!clip_model) {
            error_ = "Failed to load CLIP model: model pointer is null";
            logError(error_);
            if(callback)
                callback("Failed to load CLIP model: model pointer is null", userData);
            return false;
        }

        logInfo("CLIP model loaded successfully");
        if(callback)
            callback("CLIP model loaded successfully", userData);

        return true;
    }
    catch (const std::exception& e) {
        // Catch standard exceptions
        error_ = "Exception while loading CLIP model: " + std::string(e.what());
        logError(error_);
        if(callback)
            callback(error_.c_str(), userData);
        return false;
    }
    catch (...) {
        // Catch any other exceptions
        error_ = "Unknown exception while loading CLIP model";
        logError(error_);
        if(callback)
            callback(error_.c_str(), userData);
        return false;
    }
}


bool LlamaRuntime::createSession(int session_id) {
    if (sessions.find(session_id) != sessions.end()) {
        logError("Session already exists: " + std::to_string(session_id));
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = context_size;

    LlamaSession* new_session = new LlamaSession(std::to_string(session_id), nullptr, nullptr);
    new_session->ctx = llama_new_context_with_model(model, ctx_params);

    if (!new_session->ctx) {
        logError("Failed to create context for session " + std::to_string(session_id));
        delete new_session;
        return false;
    }

    new_session->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(new_session->smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(new_session->smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(new_session->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    sessions[session_id] = new_session;
    logInfo("Created session: " + std::to_string(session_id));
    return true;
}

bool LlamaRuntime::clearSession(int session_id) {
    auto it = sessions.find(session_id);
    if (it == sessions.end()) {
        logError("Session not found: " + std::to_string(session_id));
        return false;
    }

    it->second->clearHistory();

    logInfo("Cleared session history: " + std::to_string(session_id));
    return true;
}

bool LlamaRuntime::deleteSession(int session_id) {
    auto it = sessions.find(session_id);
    if (it == sessions.end()) {
        logError("Session not found: " + std::to_string(session_id));
        return false;
    }

    delete it->second;
    sessions.erase(it);
    logInfo("Deleted session: " + std::to_string(session_id));
    return true;
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

        // Resize formatted buffer for context size
        /*session->formatted.resize(n_ctx);*/
    }

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

    if(hasVision())
        return ::generateVision(session_id, input_prompt, callback, userData);

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
    logDebug("Messages in history: " + std::to_string(session->messages.size())+ "\n");

    // add the user input to the message list and format it
    session->messages.push_back({"user", strdup(input_prompt.c_str())});

    int new_len = llama_chat_apply_template(llama_model_chat_template(model, nullptr), session->messages.data(), session->messages.size(), true, session->formatted.data(), session->formatted.size());
    if (new_len > session->formatted.size()) {
        session->formatted.resize(new_len);
        new_len = llama_chat_apply_template(llama_model_chat_template(model, nullptr), session->messages.data(), session->messages.size(), true, session->formatted.data(), session->formatted.size());
    }
    if (new_len < 0) {
        error_ = "Error: failed to apply the chat template";
        logError(error_);
        return false;
    }

    // remove previous messages to obtain the prompt to generate the response
    std::string prompt(session->formatted.begin(), session->formatted.begin() + new_len);

    // Log tokenized prompt
    logDebug("Tokenized prompt: " + prompt+ "\n");

    // generate a response
    if (!generate(session, prompt, callback, userData))
    {
        return false;
    }

    // add the response to the messages, this is the history context used to provide llm with context in future prompts
    session->messages.push_back({"assistant", strdup(session->response.c_str())});

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

/**
 * This function executes the actual text generation using a given llama_context
 * and sampler. The response is processed and streamed via the callback.
 *
 * @param session The session for which the response is to be generated.
 * @param prompt The prompt string to generate a response for.
 * @param options Additional options for the generation process.
 * @param callback A callback function to be invoked for each generated token.
 * @param userData User data to be passed to the callback function.
 * @return True if the generation is successful, otherwise false.
 *
 * Implementation Details:
 * - The function uses a loop to generate tokens until the context is full or the generation is complete.
 * - It checks the context size and breaks the loop if the context is exceeded.
 * - Each generated token is processed and added to the session's response.
 * - The function uses a callback to stream the generated tokens.
 * - The `token_count` variable is used to prevent infinite looping.
 */
// Generate response (following gemma3-cli.cpp style)
bool LlamaRuntime::generate(LlamaSession *session, const std::string &prompt, void (*callback)(const char*, void *), void *userData) {
    if (!session) {
        error_ = "Error: Generate, session is null";
        logError(error_);
        return false;
    }

    session->response.clear();
    llama_context* ctx = session->ctx;
    llama_sampler *smpl = session->smpl;

    const bool is_first = llama_get_kv_cache_used_cells(ctx) == 0;

    // If prompt is not empty, process it first
    std::vector<llama_token> prompt_tokens;
    if (!prompt.empty()) {
        prompt_tokens = tokenizePrompt(prompt, is_first);
        if (prompt_tokens.empty()) {
            error_ = "Error: Failed to tokenize the prompt";
            logError(error_);
            return false;
        }

        logDebug("Total tokens in prompt: " + std::to_string(prompt_tokens.size())+ "\n");

        // Create a properly initialized batch manually instead of using llama_batch_get_one
        llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
        batch.n_tokens = prompt_tokens.size();

        // Set the token array
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            batch.token[i] = prompt_tokens[i];
        }

        // Set positions for tokens
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            batch.pos[i] = session->n_past + i;
            batch.n_seq_id[i] = 1;
            batch.logits[i] = false;
            // Set sequence ID pointers - assuming a single sequence ID of 0
            batch.seq_id[i][0] = 0;
        }

        // Last token should have logits
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch)) {
            error_ = "Error: failed to decode prompt";
            logError(error_);
            llama_batch_free(batch);
            return false;
        }

        // Update position counter
        session->n_past += prompt_tokens.size();

        // Free the batch
        llama_batch_free(batch);
    }

    // Generation loop
    llama_token new_token_id;
    long token_count = 0;
    while (true) {
        int n_ctx_total = llama_n_ctx(ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);

        if (n_ctx_used >= n_ctx_total - 4) {
            logError("Context size exceeded! Used: " + std::to_string(n_ctx_used) + ", Limit: " + std::to_string(n_ctx_total) + "\n");
            break;
        }

        // Sample next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
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

        std::string piece;
        piece.assign(buf, n);  // More robust than std::string(buf, n)

        if (!isValidUtf8(piece)) {  // Validate UTF-8
            logDebug("Warning: Token ID " + std::to_string(new_token_id) + " produced invalid UTF-8, skipping.");
        }
        else {
            if (callback)
                callback(piece.c_str(), userData);

            session->response += piece;
        }

        // Prepare batch for next token - use llama_batch_init to ensure arrays are properly allocated
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;

        // Set token, position, and other attributes
        batch.token[0] = new_token_id;
        batch.pos[0] = session->n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx, batch)) {
            error_ = "Error: failed to decode token";
            logError(error_);
            llama_batch_free(batch);
            return false;
        }

        // Update position counter
        session->n_past += 1;
        token_count++;

        // Free the batch
        llama_batch_free(batch);

        // Safety check to prevent infinite loops
        if (token_count > 4096) {
            logWarning("Generation exceeded maximum token count, stopping.");
            break;
        }
    }

    return true;
}


const std::string LlamaRuntime::getResponse(int session_id) {
    LlamaSession *session = getSession(session_id);
    if (session)
        return session->response;

    return std::string();
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

        ss << "Session ID: " << session->sessionId << "\n";
        ss << "Total Messages: " << session->messages.size() << "\n";
        for (size_t i = 0; i < session->messages.size(); i++) {
            size_t char_count = strlen(session->messages[i].content);
            int token_count = llama_tokenize(vocab, session->messages[i].content, char_count, nullptr, 0, true, false);
            total_char_count += char_count;
            total_token_count += token_count;

            ss << "Message " << i << " | Role: " << session->messages[i].role
               << " | Size: " << char_count << " chars, " << token_count << " tokens\n";
        }

        int used_size = total_token_count;
        int remaining_size = context_size - used_size;

        ss << "\nUsed Context Size: " << used_size << " tokens\n";
        ss << "Remaining Context Size: " << remaining_size << " tokens\n\n";
    }

    #ifdef SESSION_TEST
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
#endif

    return ss.str();
}

// Helper function to evaluate text without generating a response
bool LlamaRuntime::eval_text(LlamaSession *session, const std::string &text) {
    if (!session || !session->ctx) {
        error_ = "Error: Invalid session or context";
        logError(error_);
        return false;
    }

    logDebug("Evaluating text: " + text);

    // Get the is_first flag (true if KV cache is empty)
    const bool is_first = llama_get_kv_cache_used_cells(session->ctx) == 0;

    // Tokenize the input text
    std::vector<llama_token> tokens = tokenizePrompt(text, is_first);
    if (tokens.empty()) {
        error_ = "Error: Failed to tokenize text";
        logError(error_);
        return false;
    }

    // Create a batch for decoding with proper capacity
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

    // Fill batch with token data
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;  // Sequence ID 0
        batch.logits[i] = (i == tokens.size() - 1) ? true : false;  // Only last token needs logits
    }

    // Decode the batch
    if (llama_decode(session->ctx, batch)) {
        error_ = "Error: Failed to decode text";
        logError(error_);
        llama_batch_free(batch);
        return false;
    }

    // Update the session's past token count
    session->n_past += tokens.size();
    logDebug("Text evaluated successfully, n_past now: " + std::to_string(session->n_past));

    // Clean up
    llama_batch_free(batch);

    return true;
}

// Process image file and embed it (following gemma3-cli.cpp style)
// Process image file and embed it (following gemma3-cli.cpp style)
bool LlamaRuntime::processImageFileAndEmbed(LlamaSession *session, const std::string &image_path) {
    if (!clip_model || !session || !model) {
        error_ = "Error: CLIP model or session not initialized";
        logError(error_);
        return false;
    }

    llama_context* ctx = session->ctx;
    if (!ctx) {
        error_ = "Error: Context not initialized";
        logError(error_);
        return false;
    }

    // Get embeddings dimension
    int n_embd = llama_model_n_embd(model);
    int n_image_tokens = 256; // Default for most vision models

    // Allocate memory for embeddings
    std::vector<float> image_embd(n_image_tokens * n_embd);

    // Create image structure
    clip_image_u8* img_u8 = clip_image_u8_init();
    if (!img_u8) {
        error_ = "Error: Failed to initialize image";
        logError(error_);
        return false;
    }

    // Load image from file
    logInfo("Loading image from file: " + image_path);
    bool load_ok = clip_image_load_from_file(image_path.c_str(), img_u8);
    if (!load_ok) {
        error_ = "Error: Failed to load image from file: " + image_path;
        logError(error_);
        clip_image_u8_free(img_u8);
        return false;
    }

    // Preprocess image
    clip_image_f32_batch batch_f32;
    bool preprocess_ok = clip_image_preprocess(clip_model, img_u8, &batch_f32);
    if (!preprocess_ok) {
        error_ = "Error: Failed to preprocess image";
        logError(error_);
        clip_image_u8_free(img_u8);
        return false;
    }

    // Encode image to embeddings
    logInfo("Encoding image to embeddings");
    bool encode_ok = clip_image_batch_encode(clip_model, 4, &batch_f32, image_embd.data());
    if (!encode_ok) {
        error_ = "Error: Failed to encode image";
        logError(error_);
        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);
        return false;
    }

    clip_image_f32_batch_free(&batch_f32);
    clip_image_u8_free(img_u8);

    // Process and embed the image following gemma3-cli.cpp approach

    // 1. Add start-of-image marker
    if (eval_text(session, "<start_of_image>")) {
        error_ = "Error: Failed to process start-of-image marker";
        logError(error_);
        return false;
    }

    // 2. Disable causal attention for image processing
    llama_set_causal_attn(ctx, false);

    // 3. Create vectors for batch data
    std::vector<llama_pos> pos(n_image_tokens);
    std::vector<int32_t> n_seq_id(n_image_tokens);
    std::vector<llama_seq_id> seq_id_0(1, 0);  // Single sequence ID with value 0
    std::vector<llama_seq_id*> seq_ids(n_image_tokens + 1);
    std::vector<int8_t> logits(n_image_tokens);

    // 4. Set seq_ids last element to nullptr
    seq_ids[n_image_tokens] = nullptr;

    // 5. Initialize batch structure
    llama_batch batch = {
        /* n_tokens   = */ n_image_tokens,
        /* tokens     = */ nullptr,
        /* embd       = */ image_embd.data(),
        /* pos        = */ pos.data(),
        /* n_seq_id   = */ n_seq_id.data(),
        /* seq_id     = */ seq_ids.data(),
        /* logits     = */ logits.data(),
    };

    // 6. Fill batch arrays
    for (int i = 0; i < n_image_tokens; i++) {
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = seq_id_0.data();
        batch.logits[i] = false;
    }

    // 7. Decode image embeddings
    if (llama_decode(ctx, batch)) {
        error_ = "Error: Failed to decode image embeddings";
        logError(error_);
        return false;
    }

    // 8. Update past tokens count
    session->n_past += n_image_tokens;

    // 9. Re-enable causal attention
    llama_set_causal_attn(ctx, true);

    // 10. Add end-of-image marker
    if (eval_text(session, "<end_of_image>")) {
        error_ = "Error: Failed to process end-of-image marker";
        logError(error_);
        return false;
    }

    logInfo("Image processed and embedded successfully");
    return true;
}

// Process image from pixel data and embed it (following gemma3-cli.cpp style)
// Process image from pixel data and embed it (following gemma3-cli.cpp style)
bool LlamaRuntime::processImagePixelsAndEmbed(LlamaSession *session, const uint8_t* rgb_pixels, int width, int height) {
    if (!clip_model || !session || !model) {
        error_ = "Error: CLIP model or session not initialized";
        logError(error_);
        return false;
    }

    llama_context* ctx = session->ctx;
    if (!ctx) {
        error_ = "Error: Context not initialized";
        logError(error_);
        return false;
    }

    // Get embeddings dimension
    int n_embd = llama_model_n_embd(model);
    int n_image_tokens = 256; // Default for most vision models

    // Allocate memory for embeddings
    std::vector<float> image_embd(n_image_tokens * n_embd);

    // Create image structure
    clip_image_u8* img_u8 = clip_image_u8_init();
    if (!img_u8) {
        error_ = "Error: Failed to initialize image";
        logError(error_);
        return false;
    }

    // Build image from pixel data
    logInfo("Building image from pixels");
    clip_build_img_from_pixels(rgb_pixels, width, height, img_u8);

    // Preprocess image
    clip_image_f32_batch batch_f32;
    bool preprocess_ok = clip_image_preprocess(clip_model, img_u8, &batch_f32);
    if (!preprocess_ok) {
        error_ = "Error: Failed to preprocess image";
        logError(error_);
        clip_image_u8_free(img_u8);
        return false;
    }

    // Encode image to embeddings
    logInfo("Encoding image to embeddings");
    bool encode_ok = clip_image_batch_encode(clip_model, 4, &batch_f32, image_embd.data());
    if (!encode_ok) {
        error_ = "Error: Failed to encode image";
        logError(error_);
        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);
        return false;
    }

    clip_image_f32_batch_free(&batch_f32);
    clip_image_u8_free(img_u8);

    // Process and embed the image following gemma3-cli.cpp approach

    // 1. Add start-of-image marker
    if (eval_text(session, "<start_of_image>")) {
        error_ = "Error: Failed to process start-of-image marker";
        logError(error_);
        return false;
    }

    // 2. Disable causal attention for image processing
    llama_set_causal_attn(ctx, false);

    // 3. Create vectors for batch data
    std::vector<llama_pos> pos(n_image_tokens);
    std::vector<int32_t> n_seq_id(n_image_tokens);
    std::vector<llama_seq_id> seq_id_0(1, 0);  // Single sequence ID with value 0
    std::vector<llama_seq_id*> seq_ids(n_image_tokens + 1);
    std::vector<int8_t> logits(n_image_tokens);

    // 4. Set seq_ids last element to nullptr
    seq_ids[n_image_tokens] = nullptr;

    // 5. Initialize batch structure
    llama_batch batch = {
        /* n_tokens   = */ n_image_tokens,
        /* tokens     = */ nullptr,
        /* embd       = */ image_embd.data(),
        /* pos        = */ pos.data(),
        /* n_seq_id   = */ n_seq_id.data(),
        /* seq_id     = */ seq_ids.data(),
        /* logits     = */ logits.data(),
    };

    // 6. Fill batch arrays
    for (int i = 0; i < n_image_tokens; i++) {
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = seq_id_0.data();
        batch.logits[i] = false;
    }

    // 7. Decode image embeddings
    if (llama_decode(ctx, batch)) {
        error_ = "Error: Failed to decode image embeddings";
        logError(error_);
        return false;
    }

    // 8. Update past tokens count
    session->n_past += n_image_tokens;

    // 9. Re-enable causal attention
    llama_set_causal_attn(ctx, true);

    // 10. Add end-of-image marker
    if (eval_text(session, "<end_of_image>")) {
        error_ = "Error: Failed to process end-of-image marker";
        logError(error_);
        return false;
    }

    logInfo("Image processed and embedded successfully");
    return true;
}

// Generate response with image from file
// Generate response with image from file (following gemma3-cli.cpp style)
bool LlamaRuntime::generateResponseWithImageFile(int session_id, const std::string &input_prompt,
                                                 const std::string &image_path,
                                                 void (*callback)(const char*, void *userData),
                                                 void *userData) {
    main_vision(input_prompt.c_str(), image_path.c_str());

    return generateResponse(session_id, input_prompt, callback, userData);

    // Get the session
    LlamaSession *session = getSession(session_id);
    if (!session) {
        error_ = "Error: Session is invalid.";
        logError(error_);
        return false;
    }

    // Check if models are loaded
    if (!model || !clip_model) {
        error_ = "Error: Models not loaded.";
        logError(error_);
        return false;
    }

    session->response.clear();

    // Following gemma3-cli.cpp is_single_turn approach

    // 1. Add system prompt if needed
    if (llama_get_kv_cache_used_cells(session->ctx) == 0) {
        if (eval_text(session, "<bos>")) {
            return false;
        }
    }

    // 2. Start user turn
    if (eval_text(session, "<start_of_turn>user\n")) {
        return false;
    }

    // 3. Process the image
    if (!processImageFileAndEmbed(session, image_path)) {
        return false;
    }

    // 4. Add user prompt if provided
    if (!input_prompt.empty()) {
        if (eval_text(session, input_prompt)) {
            return false;
        }
    }

    // 5. End user turn and start model turn
    if (eval_text(session, "<end_of_turn><start_of_turn>model\n")) {
        return false;
    }

    // 6. Generate the response
    if (!generateVision(session, input_prompt, callback, userData)) {
        return false;
    }

    // 7. End model turn
    if (eval_text(session, "<end_of_turn>")) {
        return false;
    }

    // Store the messages for history context
    session->messages.push_back({"user", strdup(input_prompt.c_str())});
    session->messages.push_back({"assistant", strdup(session->response.c_str())});

    return true;
}

// Generate response with image from pixel data (following gemma3-cli.cpp style)
bool LlamaRuntime::generateResponseWithImagePixels(int session_id, const std::string &input_prompt,
                                                   const uint8_t* rgb_pixels, int width, int height,
                                                   void (*callback)(const char*, void *userData),
                                                   void *userData) {
    // Get the session
    LlamaSession *session = getSession(session_id);
    if (!session) {
        error_ = "Error: Session is invalid.";
        logError(error_);
        return false;
    }

    // Check if models are loaded
    if (!model || !clip_model) {
        error_ = "Error: Models not loaded.";
        logError(error_);
        return false;
    }

    session->response.clear();

    // Following gemma3-cli.cpp is_single_turn approach

    // 1. Add system prompt if needed
    if (llama_get_kv_cache_used_cells(session->ctx) == 0) {
        if (eval_text(session, "<bos>")) {
            return false;
        }
    }

    // 2. Start user turn
    if (eval_text(session, "<start_of_turn>user\n")) {
        return false;
    }

    // 3. Process the image
    if (!processImagePixelsAndEmbed(session, rgb_pixels, width, height)) {
        return false;
    }

    // 4. Add user prompt if provided
    if (!input_prompt.empty()) {
        if (eval_text(session, input_prompt)) {
            return false;
        }
    }

    // 5. End user turn and start model turn
    if (eval_text(session, "<end_of_turn><start_of_turn>model\n")) {
        return false;
    }

    // 6. Generate the response
    if (!generateVision(session, "", callback, userData)) {
        return false;
    }

    // 7. End model turn
    if (eval_text(session, "<end_of_turn>")) {
        return false;
    }

    // Store the messages for history context
    session->messages.push_back({"user", strdup(input_prompt.c_str())});
    session->messages.push_back({"assistant", strdup(session->response.c_str())});

    return true;
}


// Generate response (following gemma3-cli.cpp style)
// Generate response (following gemma3-cli.cpp style)
bool LlamaRuntime::generateVision(LlamaSession *session, const std::string &prompt, void (*callback)(const char*, void *), void *userData) {
    if (!session || !session->ctx) {
        error_ = "Error: Invalid session or context";
        logError(error_);
        return false;
    }

    session->response.clear();
    llama_context* ctx = session->ctx;

    // Process initial prompt if provided
    if (!prompt.empty()) {
        if (eval_text(session, prompt)) {
            return false;
        }
    }

    // Make sure vocab is valid
    if (!vocab) {
        vocab = llama_model_get_vocab(model);
    }

    // Generation parameters
    int n_predict = 4096; // Maximum tokens to generate
    float temp = 0.8f;    // Temperature
    float topp = 0.95f;   // Top-p sampling

    for (int i = 0; i < n_predict; i++) {
        // Check context size limits
        int n_ctx_total = llama_n_ctx(ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);

        if (n_ctx_used >= n_ctx_total - 4) {
            logInfo("Context size limit reached");
            break;
        }

        // Get logits for the last token
        float* logits = llama_get_logits(ctx);
        if (!logits) {
            error_ = "Error: Failed to get logits";
            logError(error_);
            return false;
        }

        // Get vocabulary size
        //int n_vocab = llama_vocab_size(vocab);
        // Replace this line:
        //int n_vocab = ::llama_vocab_size(vocab);

        // With one of these lines, depending on which is available in your llama.cpp version:
        int n_vocab = ::llama_n_vocab(vocab);  // If available
        // OR
        //int n_vocab = llama_vocab_n_vocab(vocab);  // If available

        // Simple token sampling implementation
        llama_token token_id = -1;
        try {
            // Apply temperature to logits
            std::vector<float> probs(n_vocab);
            float max_logit = -INFINITY;

            // Find max logit for numerical stability
            for (int token_idx = 0; token_idx < n_vocab; token_idx++) {
                if (logits[token_idx] > max_logit) {
                    max_logit = logits[token_idx];
                }
            }

            // Compute softmax with temperature
            float sum = 0.0f;
            for (int token_idx = 0; token_idx < n_vocab; token_idx++) {
                probs[token_idx] = expf((logits[token_idx] - max_logit) / temp);
                sum += probs[token_idx];
            }

            // Normalize
            for (int token_idx = 0; token_idx < n_vocab; token_idx++) {
                probs[token_idx] /= sum;
            }

            // Apply top-p sampling
            std::vector<std::pair<float, llama_token>> sorted_probs;
            for (int token_idx = 0; token_idx < n_vocab; token_idx++) {
                sorted_probs.push_back({probs[token_idx], token_idx});
            }

            // Sort by probability (descending)
            std::sort(sorted_probs.begin(), sorted_probs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            // Apply top-p filtering
            float cumsum = 0.0f;
            for (size_t i = 0; i < sorted_probs.size(); i++) {
                cumsum += sorted_probs[i].first;
                if (cumsum > topp) {
                    // Truncate the vector
                    sorted_probs.resize(i + 1);
                    break;
                }
            }

            // Renormalize
            sum = 0.0f;
            for (const auto& p : sorted_probs) {
                sum += p.first;
            }
            for (auto& p : sorted_probs) {
                p.first /= sum;
            }

            // Sample a token
            float r = (float)rand() / RAND_MAX;
            cumsum = 0.0f;
            token_id = sorted_probs[sorted_probs.size() - 1].second; // Default to last token

            for (const auto& p : sorted_probs) {
                cumsum += p.first;
                if (r < cumsum) {
                    token_id = p.second;
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            error_ = "Exception during sampling: " + std::string(e.what());
            logError(error_);
            return false;
        }

        // Check for end of generation
        if (token_id < 0 || llama_vocab_is_eog(vocab, token_id)) {
            logInfo("End of generation token encountered");
            break;
        }

        // Convert token to text piece
        std::string piece;
        try {
            char buf[256] = {0};
            int n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                error_ = "Error: Failed to convert token to piece";
                logError(error_);
                continue;  // Try to continue with next token
            }
            piece = std::string(buf, n);
        }
        catch (...) {
            error_ = "Exception converting token to text";
            logError(error_);
            continue;  // Try to continue with next token
        }

        // Output the token
        if (callback) {
            callback(piece.c_str(), userData);
        }
        session->response += piece;

        // Process the token to update context
        try {
            // Create batch for the token
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0] = token_id;
            batch.pos[0] = session->n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            // Decode the token
            if (llama_decode(ctx, batch)) {
                error_ = "Error: Failed to decode token";
                logError(error_);
                llama_batch_free(batch);
                return false;
            }

            // Update past position
            session->n_past++;

            // Clean up
            llama_batch_free(batch);
        }
        catch (...) {
            error_ = "Exception processing token";
            logError(error_);
            return false;
        }
    }

    return true;
}

// Helper function to get token text (similar to common_token_to_piece)
std::string LlamaRuntime::common_token_to_piece(llama_token token) {
    std::string result;

    // Get max possible length
    int n = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
    if (n <= 0) {
        return result;
    }

    // Allocate buffer and get actual text
    result.resize(n + 1); // +1 for null terminator
    int actual_len = llama_token_to_piece(vocab,
                                          token, &result[0], result.size(), 0, true);

    if (actual_len > 0) {
        result.resize(actual_len); // Remove null terminator from string
        return result;
    }

    return "";
}
