#include "LlamaEngine.h"
#include <iostream>
#include <vector>
#include <string>

#include "llama.h"
#include "LlamaRuntime.h"

static LlamaRuntime *runtimeContext = nullptr;

LlamaEngine_API bool loadModel(const char* modelPath,
    struct ModelParameter* params, size_t paramCount,
    void (*callback)(const char*)) {

    if(runtimeContext){
        std::string message = "Loading model already loaded\n";
        if (callback)
            callback(message.c_str());
        return true;
    }

    std::string message = "Loading model: " + std::string(modelPath);
    if (callback)
        callback(message.c_str());

    runtimeContext = new LlamaRuntime;
    runtimeContext->setModelPath(modelPath);

    // Iterate over parameters and print values
    for (size_t i = 0; i < paramCount; ++i) {
        std::string paramName(params[i].key);

        if (params[i].type == PARAM_FLOAT) {
            float fval = *(float*)params[i].value;
            std::string paramMessage = paramName + ": " + std::to_string(fval);
            if (callback) {
                callback(paramMessage.c_str());
            }

            if(paramName == "temperature")
                runtimeContext->setTemperature(fval);
            else if(paramName == "repetition_penalty")
                runtimeContext->setRepetitionPenalty(fval);
            else if(paramName == "top_P")
                runtimeContext->setTopP(fval);
            else if(paramName == "top_k")
                runtimeContext->setTopK(fval);
            else
            {
                std::string paramWarning = "Unused parameter "+paramName;
                if (callback) {
                    callback(paramWarning.c_str());
                }
            }
        }
        else if (params[i].type == PARAM_INT) {
            int ival = *(int*)params[i].value;

            std::string paramMessage = paramName + ": " + std::to_string(ival);
            if (callback) {
                callback(paramMessage.c_str());
            }

            if(paramName == "context_size"){
                runtimeContext->setContextSize(ival);
            }
            else {
                std::string paramWarning = "Unused parameter "+paramName;
                if (callback) {
                    callback(paramWarning.c_str());
                }
            }

        }
        else if (params[i].type == PARAM_STRING) {
            std::string paramMessage = std::string(params[i].key) + ": " + (char*)params[i].value;
            if (callback) {
                callback(paramMessage.c_str());
            }
        }
        else {
            std::string paramMessage = std::string(params[i].key) + ": Unknown Type";
            if (callback) {
                callback(paramMessage.c_str());
            }
        }
    }

    runtimeContext->setLogCallback([callback](const std::string& msg) {
        callback(msg.c_str());
    });

    if(!runtimeContext->loadModel()){
        delete runtimeContext;
        runtimeContext = nullptr;
        return false;
    }
    return true;
}

LlamaEngine_API bool generateResponse(const char* prompt, void (*callback)(const char*, void *userData), void *userData) {
    
    return runtimeContext->generateResponse(prompt, callback, userData);
}

LlamaEngine_API char* parseGGUF(const char* filepath, GGUFAttributeCallback callback, void (*messageCallback)(const char* message), void *user_data) {
    // Parse GGUF metadata using runtimeContext
    GGUFMetadata guffMetadata = LlamaRuntime::parseGGUF(filepath, messageCallback);

    // Initialize LlmMetadata structure to hold the parsed data
    static LlmMetadata metadata; // Static so it persists after the function returns

    // Extract model name (if available)
    auto nameEntry = guffMetadata.entries.find("model_name");
    metadata.name = (nameEntry != guffMetadata.entries.end() && nameEntry->second.type == TYPE_STRING)
        ? nameEntry->second.svalue.c_str()
        : "UnknownModel";  // Default fallback

    // Process attributes and invoke the callback if provided
    for (const auto& entry : guffMetadata.entries) {
        // Pass the appropriate pointer based on the attribute type
        if (callback) {
            if (entry.second.type == TYPE_UINT32) {
                callback(entry.first.c_str(), entry.second.type, const_cast<void*>(static_cast<const void*>(&entry.second.ivalue)), user_data);
            }
            else if (entry.second.type == TYPE_STRING) {
                callback(entry.first.c_str(), entry.second.type, const_cast<void*>(static_cast<const void*>(entry.second.svalue.c_str())), user_data);
            }
            // Handle other types as needed
        }
    }

    // Return the model name as a char* (ensure it's a valid pointer)
    return const_cast<char*>(metadata.name);
}


