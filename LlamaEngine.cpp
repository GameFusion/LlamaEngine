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
        return true;
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
 * Generates a response based on the given prompt.
 *
 * @param prompt Input prompt string.
 * @param streamCallback Function pointer to receive the response in token chunks.
 * @param finalCallback Function pointer to receive the full final response.
 * @param userData Custom user data passed to the callback.
 * @return True if the response is generated successfully, false otherwise.
 */
LlamaEngine_API bool generateResponse(const char* prompt,
                                      void (*streamCallback)(const char*, void* userData),
                                      void (*finalCallback)(const char*, void* userData),
                                      void* userData) {
    
    bool ret = runtimeContext->generateResponse(prompt, streamCallback, userData);
    if(ret && finalCallback)
        finalCallback(runtimeContext->getResponse().c_str(), userData);

    return ret;
}

/**
 * Get the latest complete response.
 * @return Returns the complete latest generated response.
 */
const char* getLastResponse() {
    return runtimeContext->getResponse().c_str();
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
