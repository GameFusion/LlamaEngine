#include "LlamaEngine.h"
#include <iostream>
#include <vector>
#include <string>

#include "llama.h"
#include "LlamaRuntime.h"

// Global pointer to the runtime context
static LlamaRuntime *runtimeContext = nullptr;

/**
 * Loads a machine learning model with specified parameters.
 *
 * @param modelPath Path to the model file.
 * @param params Array of model parameters.
 * @param paramCount Number of parameters.
 * @param callback Function pointer for logging messages.
 * @return True if the model is successfully loaded, false otherwise.
 */
LlamaEngine_API bool loadModel(const char* modelPath,
    struct ModelParameter* params, size_t paramCount,
    void (*callback)(const char*)) {

    // Check if a model is already loaded
    if(runtimeContext){
        std::string message = "Loading model already loaded\n";
        if (callback)
            callback(message.c_str());
        //return true;
        delete runtimeContext;
        runtimeContext = nullptr;
    }

    std::string message = "Loading model: " + std::string(modelPath);
    if (callback)
        callback(message.c_str());

    // Initialize runtime context
    runtimeContext = new LlamaRuntime;
    runtimeContext->setModelPath(modelPath);

    // Process parameters
    for (size_t i = 0; i < paramCount; ++i) {
        std::string paramName(params[i].key);

        if (params[i].type == PARAM_FLOAT) {
            float fval = *(float*)params[i].value;
            std::string paramMessage = paramName + ": " + std::to_string(fval);
            if (callback)
                callback(paramMessage.c_str());

             // Set runtime parameters based on recognized names
            if(paramName == "temperature")
                runtimeContext->setTemperature(fval);
            else if(paramName == "repetition_penalty")
                runtimeContext->setRepetitionPenalty(fval);
            else if(paramName == "top_P")
                runtimeContext->setTopP(fval);
            else if(paramName == "top_k")
                runtimeContext->setTopK(fval);
            else if (callback)
                callback(("Unused parameter: " + paramName).c_str());
        }
        else if (params[i].type == PARAM_INT) {
            int ival = *(int*)params[i].value;

            std::string paramMessage = paramName + ": " + std::to_string(ival);
            if (callback)
                callback(paramMessage.c_str());

            if(paramName == "context_size")
                runtimeContext->setContextSize(ival);
            else if (callback)
                callback((paramName + ": Unknown Type").c_str());
        }
        else if (params[i].type == PARAM_STRING) {
             if (callback)
                callback((paramName + ": " + (char*)params[i].value).c_str());
        }
        else if (callback)
            callback((paramName + ": Unknown Type").c_str());
    }

    // Set logging callback
    runtimeContext->setLogCallback([callback](const std::string& msg) {
        if (callback)
            callback(msg.c_str());
    });

    // Load the model and check success
    if(!runtimeContext->loadModel()){
        delete runtimeContext;
        runtimeContext = nullptr;
        return false;
    }
    return true;
}

/**
 * Loads a CLIP model for image processing.
 *
 * @param clipModelPath Path to the CLIP model file.
 * @param callback Function pointer for logging messages.
 * @return True if the model is successfully loaded, false otherwise.
 */
LlamaEngine_API bool loadClipModel(const char* clipModelPath, void (*callback)(const char*, void *userData), void *userData) {
    if (!runtimeContext) {
        if (callback)
            callback("Error: Runtime context is not initialized. Load model first.", userData);
        return false;
    }

    std::string message = "Loading CLIP model: " + std::string(clipModelPath);
    if (callback)
        callback(message.c_str(), userData);

    return runtimeContext->loadClipModel(clipModelPath, callback, userData);
}

/**
 * Checks if a vision model (CLIP) is loaded.
 *
 * @return True if the vision model is loaded, false otherwise.
 */
LlamaEngine_API bool isVisionModelLoaded() {
    if (!runtimeContext) {
        return false;
    }
    return runtimeContext->isVisionModelLoaded();
}

/**
 * @brief Creates a new session and returns a session UUID.
 *
 * @return A dynamically allocated UUID string. Caller must free the memory.
 */
LlamaEngine_API bool createSession(int sessionId) {
    return runtimeContext->createSession(sessionId);
}

/**
 * @brief Clears the context history for a specific session.
 *
 * @param sessionUuid The UUID of the session to clear.
 * @return True if successful, false if session does not exist.
 */
LlamaEngine_API bool clearSession(int sessionId) {
    return runtimeContext->clearSession(sessionId);
}

/**
 * @brief Deletes a session and frees associated resources.
 *
 * @param sessionUuid The UUID of the session to delete.
 * @return True if the session was successfully deleted, false otherwise.
 */
LlamaEngine_API bool deleteSession(int sessionId) {
    return runtimeContext->deleteSession(sessionId);
}


/**
 * @brief Generates a response for the specified session using the given prompt.
 *
 * This function retrieves the session identified by `sessionID` and uses
 * its associated context and sampler to generate a response. The response
 * is streamed through `streamCallback` in chunks and, if successful, the
 * complete response is passed to `finalCallback`.
 *
 * @param sessionID The ID of the session to use for generating the response.
 * @param prompt Input prompt string.
 * @param streamCallback Function pointer to receive the response in token chunks.
 * @param finalCallback Function pointer to receive the full final response (optional).
 * @param userData Custom user data passed to both callbacks.
 * @return True if the response is generated successfully, false otherwise.
 *
 * @note If the specified session does not exist, the function may return false.
 *       Ensure a valid session is created before calling this function.
 */
LlamaEngine_API bool generateResponse(int sessionID,
                                      const char* prompt,
                                      void (*streamCallback)(const char*, void* userData),
                                      void (*finalCallback)(const char*, void* userData),
                                      void* userData) {
    if (!runtimeContext) {
        if (streamCallback)
            streamCallback("Error: Runtime context is not initialized.", userData);
        return false;
    }
    bool ret = runtimeContext->generateResponse(sessionID, prompt, streamCallback, userData);
    if(ret && finalCallback)
        finalCallback(runtimeContext->getResponse(sessionID).c_str(), userData);

    return ret;
}

/**
 * @brief Generates a response for the specified session using the given prompt and image file.
 *
 * This function allows multimodal generation by processing both text and image inputs.
 * The image is loaded from a file, processed through the CLIP model, and embedded
 * into the prompt context.
 *
 * @param sessionID The ID of the session to use for generating the response.
 * @param prompt Input prompt string (can be empty).
 * @param imagePath Path to the image file.
 * @param streamCallback Function pointer to receive the response in token chunks.
 * @param finalCallback Function pointer to receive the full final response (optional).
 * @param userData Custom user data passed to both callbacks.
 * @return True if the response is generated successfully, false otherwise.
 */
LlamaEngine_API bool generateResponseWithImageFile(int sessionID,
                                                   const char* prompt,
                                                   const char* imagePath,
                                                   void (*streamCallback)(const char*, void* userData),
                                                   void (*finalCallback)(const char*, void* userData),
                                                   void* userData) {
    // Add debug print to verify function entry
    fprintf(stderr, "Entering generateResponseWithImageFile\n");

    if (!runtimeContext) {
        fprintf(stderr, "Error: Runtime context is null\n");
        if (streamCallback)
            streamCallback("Error: Runtime context is not initialized.", userData);
        return false;
    }

    fprintf(stderr, "Checking vision model\n");

    // Check if vision model is loaded
    bool visionLoaded = false;
    try {
        visionLoaded = runtimeContext->isVisionModelLoaded();
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in isVisionModelLoaded: %s\n", e.what());
        if (streamCallback)
            streamCallback("Error: Exception checking vision model.", userData);
        return false;
    } catch (...) {
        fprintf(stderr, "Unknown exception in isVisionModelLoaded\n");
        if (streamCallback)
            streamCallback("Error: Unknown exception checking vision model.", userData);
        return false;
    }

    if (!visionLoaded) {
        fprintf(stderr, "Vision model not loaded\n");
        if (streamCallback)
            streamCallback("Error: Vision model (CLIP) is not loaded.", userData);
        return false;
    }

    // Convert prompt and image path to std::string
    std::string promptStr = prompt ? prompt : "";
    std::string imagePathStr = imagePath ? imagePath : "";

    fprintf(stderr, "Prompt: %s\n", promptStr.c_str());
    fprintf(stderr, "Image path: %s\n", imagePathStr.c_str());

    if (imagePathStr.empty()) {
        fprintf(stderr, "Empty image path\n");
        if (streamCallback)
            streamCallback("Error: Invalid image path.", userData);
        return false;
    }

    // Check if file exists
    FILE* testFile = fopen(imagePathStr.c_str(), "rb");
    if (!testFile) {
        fprintf(stderr, "Image file not found: %s\n", imagePathStr.c_str());
        if (streamCallback)
            streamCallback("Error: Image file not found.", userData);
        return false;
    }
    fclose(testFile);

    fprintf(stderr, "Calling runtime->generateResponseWithImageFile\n");

    // Call the method with try-catch to handle exceptions
    bool ret = false;
    try {
        ret = runtimeContext->generateResponseWithImageFile(
            sessionID,
            promptStr,
            imagePathStr,
            streamCallback,
            userData
            );
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in generateResponseWithImageFile: %s\n", e.what());
        if (streamCallback)
            streamCallback("Error: Exception generating response.", userData);
        return false;
    } catch (...) {
        fprintf(stderr, "Unknown exception in generateResponseWithImageFile\n");
        if (streamCallback)
            streamCallback("Error: Unknown exception generating response.", userData);
        return false;
    }

    fprintf(stderr, "Result: %d\n", ret);

    if (ret && finalCallback) {
        // Get response with try-catch
        try {
            const std::string response = runtimeContext->getResponse(sessionID);
            finalCallback(response.c_str(), userData);
        } catch (const std::exception& e) {
            fprintf(stderr, "Exception in getResponse: %s\n", e.what());
            if (streamCallback)
                streamCallback("Error: Exception getting final response.", userData);
            return false;
        } catch (...) {
            fprintf(stderr, "Unknown exception in getResponse\n");
            if (streamCallback)
                streamCallback("Error: Unknown exception getting final response.", userData);
            return false;
        }
    }

    return ret;
}

/**
 * @brief Generates a response for the specified session using the given prompt and image pixels.
 *
 * This function allows multimodal generation using raw pixel data. The pixels are processed
 * through the CLIP model and embedded into the prompt context.
 *
 * @param sessionID The ID of the session to use for generating the response.
 * @param prompt Input prompt string (can be empty).
 * @param rgbPixels The RGB pixel data (interleaved R,G,B values).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param streamCallback Function pointer to receive the response in token chunks.
 * @param finalCallback Function pointer to receive the full final response (optional).
 * @param userData Custom user data passed to both callbacks.
 * @return True if the response is generated successfully, false otherwise.
 */
LlamaEngine_API bool generateResponseWithImagePixels(int sessionID,
                                                     const char* prompt,
                                                     const unsigned char* rgbPixels,
                                                     int width, int height,
                                                     void (*streamCallback)(const char*, void* userData),
                                                     void (*finalCallback)(const char*, void* userData),
                                                     void* userData) {
    if (!runtimeContext) {
        if (streamCallback)
            streamCallback("Error: Runtime context is not initialized.", userData);
        return false;
    }

    if (!runtimeContext->isVisionModelLoaded()) {
        if (streamCallback)
            streamCallback("Error: Vision model (CLIP) is not loaded.", userData);
        return false;
    }

    if (!rgbPixels || width <= 0 || height <= 0) {
        if (streamCallback)
            streamCallback("Error: Invalid image data or dimensions.", userData);
        return false;
    }

    // Convert prompt to std::string
    std::string promptStr = prompt ? prompt : "";

    bool ret = runtimeContext->generateResponseWithImagePixels(
        sessionID,
        promptStr,
        rgbPixels,
        width,
        height,
        streamCallback,
        userData
        );

    if (ret && finalCallback) {
        finalCallback(runtimeContext->getResponse(sessionID).c_str(), userData);
    }

    return ret;
}

/**
 * Get the latest complete response.
 * @return Returns the complete latest generated response.
 */
LlamaEngine_API const char* getLastResponse() {
    const int defaultSession = 0;
    return runtimeContext->getResponse(defaultSession).c_str();
}

LlamaEngine_API void getContextInfo(void (*callback)(const char*info, void*userData), void* userData){
    if (!runtimeContext) {
        if (callback)
            callback("Error: Runtime context is not initialized.", userData);
        return;
    }

    std::string result = runtimeContext->getContextInfo();
    callback(result.c_str(), userData);
}

/**
 * Parses GGUF metadata from a model file.
 *
 * @param filepath Path to the GGUF model file.
 * @param callback Function pointer to process extracted attributes.
 * @param messageCallback Function pointer for logging messages.
 * @param user_data Custom user data for the callback.
 * @return Pointer to the model name as a C-style string.
 */
LlamaEngine_API char* parseGGUF(const char* filepath, GGUFAttributeCallback callback, void (*messageCallback)(const char* message), void *user_data) {
    // Parse GGUF metadata using runtimeContext
    GGUFMetadata guffMetadata = LlamaRuntime::parseGGUF(filepath, messageCallback);

    // Initialize LlmMetadata structure to hold the parsed data
    // Todo, use a context or system to make this concurent/thread safe
    static LlmMetadata metadata; // Static so it persists after the function returns
    metadata = LlmMetadata(); // clear metadata

    // Extract model name or use default
    auto nameEntry = guffMetadata.entries.find("model_name");
    metadata.name = (nameEntry != guffMetadata.entries.end() && nameEntry->second.type == TYPE_STRING)
        ? nameEntry->second.svalue.c_str()
        : "UnknownModel";  // Default fallback

    // Process extracted metadata attributes and invoke the callback if provided
    for (const auto& entry : guffMetadata.entries) {
        // Pass the appropriate pointer based on the attribute type
        if (callback) {
            if (entry.second.type == TYPE_UINT32) {
                callback(entry.first.c_str(), entry.second.type, const_cast<void*>(static_cast<const void*>(&entry.second.ivalue)), user_data);
            }
            else if (entry.second.type == TYPE_STRING) {
                callback(entry.first.c_str(), entry.second.type, const_cast<void*>(static_cast<const void*>(entry.second.svalue.c_str())), user_data);
            }
            // Handle additional and future types here
        }
    }

    // Return the model name as a char*
    return const_cast<char*>(metadata.name);
}

