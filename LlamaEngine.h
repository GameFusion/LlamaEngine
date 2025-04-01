#ifndef LlamaEngine_h
#define LlamaEngine_h

#include "GGUFMetadata.h"

// -------------------------------------------------------------------------------------
// Define export/import macros for different platforms
// -------------------------------------------------------------------------------------
#ifdef _WIN32
    #ifdef LlamaEngine_EXPORTS
        #define LlamaEngine_API __declspec(dllexport)  // Export symbols when building DLL
    #else
        #define LlamaEngine_API __declspec(dllimport)  // Import symbols when using DLL
    #endif
#elif __APPLE__
    #ifdef LlamaEngine_EXPORTS
        #define LlamaEngine_API __attribute__((visibility("default")))  // Export symbols for macOS
    #else
        #define LlamaEngine_API
    #endif
#else
    // Linux and other platforms, no special export directive is required
    #define LlamaEngine_API
#endif

// -------------------------------------------------------------------------------------
// C-compatible structures for model metadata
// -------------------------------------------------------------------------------------

/**
 * @brief Structure representing metadata information about an LLM (Large Language Model).
 */
struct LlmMetadata {
    const char* name;          ///< Name of the model
    const char** attributes;   ///< Array of C-strings representing model attributes
    size_t attribute_count;    ///< Number of attributes in the model
};

/**
 * @brief Enumeration for different types of model parameters.
 */
typedef enum {
    PARAM_FLOAT,   ///< Floating-point parameter (e.g., temperature)
    PARAM_INT,     ///< Integer parameter (e.g., max token count)
    PARAM_STRING,  ///< String parameter (e.g., model name)
    PARAM_UNKNOWN  ///< Unknown or uninitialized parameter type
} ParamType;

/**
 * @brief Represents a single model parameter, used when configuring the model.
 */
struct ModelParameter {
    const char* key;   ///< Name of the parameter (e.g., "temperature")
    ParamType type;    ///< Type of the parameter (float, int, string, etc.)
    void* value;       ///< Pointer to the actual value of the parameter
};

// -------------------------------------------------------------------------------------
// C-compatible API functions
// -------------------------------------------------------------------------------------

extern "C" {

/**
 * @brief Loads a machine learning model with the specified parameters.
 *
 * @param backendType The type of backend to use (e.g., "CPU", "CUDA").
 * @param params Pointer to an array of model parameters.
 * @param paramCount Number of parameters in the array.
 * @param callback Optional callback function for logging messages.
 * @return True if the model is successfully loaded, false otherwise.
 */
LlamaEngine_API bool loadModel(const char* backendType,
                               struct ModelParameter* params, size_t paramCount,
                               void (*callback)(const char*) = nullptr);

LlamaEngine_API bool loadClipModel(const char* clipModelPath, void (*callback)(const char*, void *userData), void *userData);
LlamaEngine_API bool isVisionModelLoaded();

/**
 * @brief Creates a new session and returns a session UUID.
 *
 * @return A dynamically allocated UUID string. Caller must free the memory.
 */
LlamaEngine_API bool createSession(int sessionId);

/**
 * @brief Clears the context history for a specific session.
 *
 * @param sessionUuid The UUID of the session to clear.
 * @return True if successful, false if session does not exist.
 */
LlamaEngine_API bool clearSession(int sessionId);

/**
 * @brief Deletes a session and frees associated resources.
 *
 * @param sessionUuid The UUID of the session to delete.
 * @return True if the session was successfully deleted, false otherwise.
 */
LlamaEngine_API bool deleteSession(int sessionId);

/**
 * @brief Generates a response from the model for a given session and prompt.
 *
 * This function retrieves the session identified by `sessionId`, ensuring that
 * the associated context and sampler are used. It processes the input prompt
 * and generates a response, invoking callback functions to handle streaming
 * and final output.
 *
 * @param sessionId The ID of the session to use for generating the response.
 * @param prompt Input text for the model to generate a response.
 * @param streamCallback Function to handle generated response data as it streams.
 * @param finalCallback Function to handle the final generated response.
 * @param userData Custom user data pointer passed to both callbacks.
 * @return True if the response was successfully generated, false otherwise.
 *
 * @note If the specified session does not exist, the function will return false.
 *       Ensure that a valid session is created before calling this function.
 */
LlamaEngine_API bool generateResponse(int sessionId,
                                      const char* prompt,
                                      void (*streamCallback)(const char*, void* userData),
                                      void (*finalCallback)(const char*, void* userData),
                                      void* userData);

// Vision capabilities
LlamaEngine_API bool generateResponseWithImageFile(int sessionID, const char* prompt,
                                                   const char* imagePath,
                                                   void (*streamCallback)(const char*, void* userData),
                                                   void (*finalCallback)(const char*, void* userData),
                                                   void* userData);

LlamaEngine_API bool generateResponseWithImagePixels(int sessionID,
                                                     const char* prompt,
                                                     const unsigned char* rgbPixels,
                                                     int width, int height,
                                                     void (*streamCallback)(const char*, void* userData),
                                                     void (*finalCallback)(const char*, void* userData),
                                                     void* userData);

LlamaEngine_API const char* getLastResponse(); // Retrieve the latest full response

LlamaEngine_API void getContextInfo(void (*callback)(const char* info, void *userData), void* userData = nullptr); // Retrieve context stats and descriptive info

/**
 * @brief Parses a GGUF file and retrieves metadata attributes via a callback.
 *
 * @param filepath Path to the GGUF file.
 * @param callback Function to process key-value attributes from the file.
 * @param messageCallback Function to handle status messages during parsing.
 * @param user_data Optional user data pointer to be passed to callbacks.
 * @return A dynamically allocated string containing parsed metadata (caller must free).
 */
typedef void (*GGUFAttributeCallback)(const char* key, GGUFType type, void* value, void* user_data);

LlamaEngine_API char* parseGGUF(const char* filepath,
                                GGUFAttributeCallback callback,
                                void (*messageCallback)(const char* message),
                                void* user_data = nullptr);


}

#endif // LlamaEngine_h
