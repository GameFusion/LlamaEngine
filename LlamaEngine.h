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

/**
 * @brief Generates a response from the model based on a given prompt.
 *
 * @param prompt Input text for the model to generate a response.
 * @param callback Function to handle generated response data.
 * @param userData Custom user data pointer passed to the callback.
 * @return True if the response was successfully generated, false otherwise.
 */
LlamaEngine_API bool generateResponse(const char* prompt,
                                      void (*streamCallback)(const char*, void* userData),
                                      void (*finalCallback)(const char*, void* userData),
                                      void* userData);

LlamaEngine_API const char* getLastResponse(); // Retrieve the latest full response

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
