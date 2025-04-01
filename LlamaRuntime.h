#ifndef LlamaRuntime_h
#define LlamaRuntime_h

#include <string>
#include <vector>
#include <functional>
#include <iostream> // Optional: fallback to console output
#include <unordered_map>

#include "llama.h"
#include "gguf.h"
#include "clip.h"

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
     * @brief Sets the file path for the model.
     * @param path Path to the model file.
     */
    void setModelPath(const std::string &path);

    // -------------------------------------------------------------------------------------
    // Model Configuration Setters
    // -------------------------------------------------------------------------------------

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

    // Vision capabilities
    bool loadClipModel(const std::string &clip_model_path, void (*callback)(const char*, void *userData), void *userData);
    bool isVisionModelLoaded() const;

    // Generate response with image from file
    bool generateResponseWithImageFile(int session_id, const std::string &input_prompt,
                                       const std::string &image_path,
                                       void (*callback)(const char*, void *userData),
                                       void *userData = nullptr);

    // Generate response with image from pixel data
    bool generateResponseWithImagePixels(int session_id, const std::string &input_prompt,
                                         const uint8_t* rgb_pixels, int width, int height,
                                         void (*callback)(const char*, void *userData),
                                         void *userData = nullptr);


    /**
     * @brief Get the full response.
     */
    const std::string getResponse(int session_id);

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

    /**
     * Clears the specified session.
     *
     * @param session_id The ID of the session to create.
     * @return True if a new session was successfully created.
     */
    bool createSession(int session_id);

    /**
     * Clears the specified session.
     *
     * @param session_id The ID of the session to clear the history.
     * @return True if the session was successfully cleared, otherwise false.
     */
    bool clearSession(int session_id);

    /**
     * Delete the specified session.
     *
     * @param session_id The ID of the session to delete.
     * @return True if the session was successfully delete.
     */
    bool deleteSession(int session_id);

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
     * @brief Loads a model from a specific path with given parameters.
     * @param modelPath Path to the model file.
     * @param ngl Number of GPU layers to use.
     * @param n_ctx Context size for the model.
     * @return True if the model loads successfully, false otherwise.
     */
    bool loadModelInternal(const std::string &modelPath, int ngl, int n_ctx);


    // -------------------------------------------------------------------------------------
    // Response Generation
    // -------------------------------------------------------------------------------------

    /**
     * Generates a response for the given session based on the provided prompt.
     *
     * @param session The session for which the response is to be generated.
     * @param prompt The prompt string to generate a response for.
     * @param options Additional options for the generation process.
     * @param callback A callback function to be invoked for each generated token.
     * @param userData User data to be passed to the callback function.
     * @return True if the generation is successful, otherwise false.
     */
    bool generate(LlamaSession *session,
                  const std::string &prompt,
                  void (*callback)(const char*, void *),
                  void *userData);

    bool generateVision(LlamaSession *session,
                  const std::string &prompt,
                  void (*callback)(const char*, void *),
                  void *userData);

    // Image processing and embedding
    bool processImageFileAndEmbed(LlamaSession *session, const std::string &image_path);
    bool processImagePixelsAndEmbed(LlamaSession *session, const uint8_t* rgb_pixels, int width, int height);
    bool eval_text(LlamaSession *session, const std::string &text);

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
    clip_ctx *clip_model = nullptr;
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
    static std::string llama_date;

    std::string error_;                       ///< Last recorded error message.

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
    std::string clipModelPath;
    float topK = 40;               ///< Limits sampling to top-K probable tokens.
    float topP = 0.95;              ///< Nucleus sampling threshold.
    float repetitionPenalty = 1.1f; ///< Penalty factor for repeated tokens.

    /**
     * @brief Callback function for handling log messages.
     */
    LogCallback logCallback;


    std::string common_token_to_piece(llama_token token);
};

#endif // LlamaRuntime_h
