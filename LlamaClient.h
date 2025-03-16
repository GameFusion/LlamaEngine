/**
 * @file LlamaClient.h
 * @brief Defines the LlamaClient class for interacting with the Llama engine.
 * @details Manages model loading, response generation, and GGUF metadata parsing.
 * @author Andreas Carlen
 * @date March 6, 2025
 */

#ifndef LlamaClient_h
#define LlamaClient_h

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <dlfcn.h> // For dynamic loading on macOS
#endif

#include "LlamaEngine.h"
#include "GGUFMetadata.h"

/**
 * @class LlamaClient
 * @brief Provides an interface to load and interact with Llama models.
 */
class LlamaClient {
public:
    /**
     * @brief Constructor for LlamaClient.
     * @param backend The backend to use (e.g., "CUDA", "CPU").
     * @param dllPath Path to the dynamic library.
     */
#ifdef _WIN32
    LlamaClient(const std::string &backend = "CUDA", const std::string& dllPath = "LlamaEngine.dll");
#elif __APPLE__
    LlamaClient(const std::string &backend = "CPU", const std::string& dllPath = "LlamaEngine.dylib");
#endif

    /**
     * @brief Destructor to clean up resources.
     */
    ~LlamaClient();

    /**
     * @brief Retrieves the backend type in use.
     * @return A string representing the backend (e.g., "CUDA").
     */
    std::string backendType();

    /**
     * @brief Retrieves the library name.
     * @return A string containing the dynamic library name.
     */
    std::string libraryName();

    /**
     * @brief Creates a new instance of LlamaClient.
     * @param backend The backend to use (default: "CUDA").
     * @param dllPath Path to the dynamic library.
     * @return A pointer to the created LlamaClient instance.
     */
    static LlamaClient* Create(const std::string &backend /*= "CUDA"*/, const std::string& dllPath /*= "LlamaEngined.dll"*/);

    /**
     * @brief Retrieves any error that occurred during the creation of LlamaClient.
     * @return A reference to the error string.
     */
    static const std::string& GetCreateError();

    /**
     * @brief Loads an LLM model.
     * @param modelName Name of the model.
     * @param params Pointer to model parameters.
     * @param paramCount Number of parameters.
     * @param callback Optional callback function for status updates.
     * @return True if the model loads successfully, false otherwise.
     */
    bool loadModel(const std::string& modelName, struct ModelParameter* params, size_t paramCount, void (*callback)(const char*) = nullptr);

    bool isModelLoaded();

    std::string getModelFile();

    bool createSession(int sessionId);
    bool  clearSession(int sessionId);
    bool deleteSession(int sessionId);

    /**
     * @brief Generates a response based on a given prompt.Using default session
     * @param prompt The input text to process.
     * @param streamCallback Callback for streaming tokens.
     * @param finishedCallback Callback for completion notification.
     * @param userData User-defined data to pass to callbacks.
     * @return True if successful, false otherwise.
     */
    bool generateResponse(const std::string& prompt,
                          void (*streamCallback)(const char* msg, void* user_data),
                          void (*finishedCallback)(const char* msg, void* user_data), void *userData);

    /**
     * @brief Generates a response from the Llama model using a given prompt.
     *
     * This function processes the input text, generating a response in token chunks
     * via the streaming callback, followed by a final response callback when complete.
     *
     * @param sessionId The unique identifier for the session.
     * @param prompt The input text prompt to process.
     * @param streamCallback Function pointer to handle streamed response tokens.
     * @param finishedCallback Function pointer to receive the full generated response.
     * @param userData Optional user-defined data passed to both callbacks.
     * @return True if the response generation was successful, false otherwise.
     */
    bool generateResponse(int sessionId, const std::string& prompt,
                          void (*streamCallback)(const char* msg, void* user_data),
                          void (*finishedCallback)(const char* msg, void* user_data), void *userData);

    std::string getContextInfo();

    /**
     * @brief Parses GGUF metadata from a file.
     * @param filepath Path to the GGUF file.
     * @param callback Callback function for processing metadata.
     * @return Parsed GGUFMetadata object.
     */
    GGUFMetadata parseGGUF(const std::string& filepath, void (*callback)(const char* message));



private:
#ifdef _WIN32
    HMODULE hDll; ///< Handle to the loaded DLL
#elif __APPLE__
    void* hDll; ///< Handle to the loaded shared library
#endif

    void LoadLibrary(const std::string& dllPath);

    /** Function pointers for dynamic linking **/
    typedef bool (*LoadModelFunc)(const char*, struct ModelParameter* params, size_t paramCount, void (*)(const char*));
    typedef bool (*GenerateResponseFunc)(int sessionId, const char*, void (*)(const char* token, void* user_data), void (*)(const char* completeResponse, void* user_data), void *userData);
    typedef const char* (*ParseGGUFFunc)(const char*, void (*)(const char* key, GGUFType type, void* data, void *userData), void (*callback)(const char* message), void *userData);
    typedef void (*GetContextInfoFunc)(void (*callback)(const char* info, void *), void*);

    typedef bool (*CreateSessionFunc)(int session_id);
    typedef bool (*ClearSessionFunc)(int session_id);
    typedef bool (*DeleteSessionFunc)(int session_id);

    LoadModelFunc loadModelFunc; ///< Function pointer for loading models
    GenerateResponseFunc generateResponseFunc; ///< Function pointer for generating responses
    ParseGGUFFunc parseGGUFFunc; ///< Function pointer for parsing GGUF metadata
    GetContextInfoFunc getContextInfoFunc;

    CreateSessionFunc createSessionFunc;
    ClearSessionFunc clearSessionFunc;
    DeleteSessionFunc deleteSessionFunc;
    /**
     * @brief Handles streaming response tokens.
     * @param response The response token received.
     */
    void responseCallback(const std::string& response);

    /**
     * @brief Handles the completion of a response.
     * @param message The final response message.
     */
    void finishedCallback(const std::string& message);

    static std::string createError; ///< Stores the last creation error message

    std::string backend; ///< Backend type (CPU, CUDA, Vulkan)
    std::string library; ///< Path to the dynamic library

    bool modelLoaded = false; // Track if the model is successfully loaded
    std::string modelPathFile; // Path file name for current model

};

#endif // LlamaClient_h
