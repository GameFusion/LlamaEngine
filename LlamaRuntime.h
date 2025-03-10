#ifndef LlamaRuntime_h
#define LlamaRuntime_h

#include <string>
#include <vector>
#include <functional>
#include <iostream> // Optional: fallback to console output
#include <unordered_map>

#include "llama.h"
#include "gguf.h"

#include "GGUFMetadata.h"

class LlamaSession;

/**
 * @class LlamaRuntime
 * @brief Handles model loading, text generation, and logging for the Llama model.
 */
class LlamaRuntime {
public:
    /**
     * @brief Constructs a new LlamaRuntime instance.
     */
    LlamaRuntime();

    /**
     * @brief Destructor that cleans up allocated resources.
     */
    ~LlamaRuntime();

    /**
     * @brief Loads a default model (if a path has been set).
     * @return True if the model loads successfully, false otherwise.
     */
    bool loadModel();

    /**
     * @brief Loads a model from a specific path with given parameters.
     * @param modelPath Path to the model file.
     * @param ngl Number of GPU layers to use.
     * @param n_ctx Context size for the model.
     * @return True if the model loads successfully, false otherwise.
     */
    bool loadModelInternal(const std::string &modelPath, int ngl, int n_ctx);

    // -------------------------------------------------------------------------------------
    // Model Configuration Setters
    // -------------------------------------------------------------------------------------

    /**
     * @brief Sets the file path for the model.
     * @param path Path to the model file.
     */
    void setModelPath(const std::string &path);

    /**
     * @brief Sets the size of the context window.
     * @param size Number of tokens the model can remember in a session.
     */
    void setContextSize(int size);

    /**
     * @brief Sets the temperature parameter for response randomness.
     * @param temp Temperature value (higher values increase randomness).
     */
    void setTemperature(float temp);

    /**
     * @brief Sets the top-K sampling parameter.
     * @param k Only the top-K most probable words are considered for sampling.
     */
    void setTopK(float k);

    /**
     * @brief Sets the top-P (nucleus) sampling parameter.
     * @param p Controls the cumulative probability cutoff for sampling.
     */
    void setTopP(float p);

    /**
     * @brief Sets the repetition penalty to reduce repetitive text generation.
     * @param penalty Penalty value (values > 1 discourage repetition).
     */
    void setRepetitionPenalty(float penalty);

    // -------------------------------------------------------------------------------------
    // Response Generation
    // -------------------------------------------------------------------------------------

    /**
     * @brief Generates a response from the model based on an input prompt for a specific session.
     *
     * This function retrieves the session identified by `session_id`, then processes
     * the input prompt using the session's context and sampler. The generated response
     * is streamed in chunks via the provided callback.
     *
     * @param session_id The ID of the session to use for generating the response.
     * @param input_prompt The text prompt provided by the user.
     * @param callback Function pointer for handling generated responses in chunks.
     * @param userData Optional user-defined data passed to the callback.
     * @return True if generation was successful, false otherwise.
     *
     * @note Ensure that the session exists before calling this function.
     */
    bool generateResponse(int session_id,
                            const std::string &input_prompt,
                            void (*callback)(const char*, void *userData),
                            void *userData);
    /**
     * @brief Get the full response.
     */
    const std::string getResponse();

    /**
     * @brief Parses a GGUF file and extracts metadata.
     * @param filepath Path to the GGUF file.
     * @param callback Callback function for logging or processing messages.
     * @return A GGUFMetadata object containing parsed metadata.
     */
    static GGUFMetadata parseGGUF(const std::string& filepath, void(*callback)(const char* message));

    // -------------------------------------------------------------------------------------
    // Context
    // -------------------------------------------------------------------------------------

    std::string getContextInfo();

    // -------------------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------------------

    /**
     * @brief Defines a logging callback function type.
     */
    using LogCallback = std::function<void(const std::string&)>;

    /**
     * @brief Sets a callback function for logging messages.
     * @param callback A function to handle log messages.
     */
    void setLogCallback(LogCallback callback);

    /**
     * @brief Logs a generic message.
     * @param message The message to log.
     */
    void logMessage(const std::string& message);

    /**
     * @brief Logs an informational message. Format message output like '[INFO] message'
     * @param infoMessage The informational message.
     */
    void logInfo(const std::string& errorMessage);

    /**
     * @brief Logs an error message. Format output type like '[ERROR] message'
     * @param errorMessage The error message.
     */
    void logError(const std::string& errorMessage);

    /**
     * @brief Logs a warning message. Format output type like '[WARNING] message'
     * @param warningMessage The warning message.
     */
    void logWarning(const std::string& warningMessage);

    /**
     * @brief Logs a debug message. Format output type like '[DEBUG] message'
     * @param debugMessage The warning message.
     */
    void logDebug(const std::string& debugMessage);

private:
    /**
     * @brief Internal method to generate a response from the model.
     *
     * This function executes the actual text generation using a given llama_context
     * and sampler. The response is processed and streamed via the callback.
     *
     * @param ctx Pointer to the llama context associated with the session.
     * @param smpl Pointer to the sampler used for text generation.
     * @param prompt The input text prompt for generation.
     * @param callback Function pointer for handling generated text chunks.
     * @param userData Optional user-defined data passed to the callback.
     * @return True if generation was successful, false otherwise.
     *
     * @note This function is called internally by `generateResponse` and requires a valid context.
     */
    bool generate(llama_context* ctx,
                  llama_sampler *smpl,
                  const std::string &prompt,
                  void (*callback)(const char*, void *),
                  void *userData);
    /**
     * @brief Tokenizes an input prompt before feeding it to the model.
     * @param prompt The text to tokenize.
     * @param is_first Indicates if it's the first input in a session.
     * @return A vector of tokenized words.
     */
    std::vector<llama_token> tokenizePrompt(const std::string &prompt, bool is_first);

    // -------------------------------------------------------------------------------------
    // Model Data Members
    // -------------------------------------------------------------------------------------

    llama_model *model = nullptr;  ///< Pointer to the loaded model.
    const llama_vocab *vocab = nullptr; ///< Pointer to model vocabulary.

    /**
     * @brief Llama model version (retrieved from git describe).
     * Run the command '$ git describe' in the llama.cpp repository to obtain this value.
     */
    static std::string llama_version;

    /**
     * @brief Date of the Llama model commit (retrieved from git log).
     * Run the command 'git log - 1 --format = % cd' in the llama.cpp repository to obtain this value.
     */

    /**
     * @brief Date of the Llama model commit (retrieved from git log).
     * Run the command '$ git log -1 --format=%cd' in the llama.cpp repository to obtain this value.
     */
    static std::string llama_date;

    std::vector<llama_chat_message> messages; ///< Stores chat messages.
    std::vector<char> formatted;              ///< Formatted message buffer.
    std::string error_;                       ///< Last recorded error message.
    std::vector<std::string> errorMessages;   ///< List of error messages.
    std::string response;                     ///< Last generated response.

    // -------------------------------------------------------------------------------------
    // Session Management
    // -------------------------------------------------------------------------------------

    /**
     * @brief Retrieves a session by its session ID.
     *
     * This function looks up the session in the session map and returns a pointer to
     * the associated LlamaSession instance if it exists.
     *
     * @param session_id The ID of the session to retrieve.
     * @return Pointer to the LlamaSession if found, nullptr otherwise.
     */
    LlamaSession *getSession(int session_id);

    /**
     * @brief A map storing active Llama sessions.
     *
     * Each session is identified by a unique session ID and contains its
     * own context and sampler for text generation.
     */
    std::unordered_map<int, LlamaSession*> sessions;

    // -------------------------------------------------------------------------------------
    // Model Configuration Parameters
    // -------------------------------------------------------------------------------------

    float temperature = 0.8f;      ///< Controls randomness in generation.
    int context_size = 4096;       ///< Number of tokens the model remembers.
    std::string modelPath;         ///< Path to the model file.
    float topK = 40;               ///< Limits sampling to top-K probable tokens.
    float topP = 1.0;              ///< Nucleus sampling threshold.
    float repetitionPenalty = 1.0f; ///< Penalty factor for repeated tokens.

    /**
     * @brief Callback function for handling log messages.
     */
    LogCallback logCallback;
};

#endif // LlamaRuntime_h
