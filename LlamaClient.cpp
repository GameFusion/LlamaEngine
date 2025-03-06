#include "LlamaClient.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <dlfcn.h>
#endif

std::string LlamaClient::createError;

LlamaClient::LlamaClient(const std::string &backendType, const std::string& dllPath) {
    backend = backendType;
    library = dllPath;

#ifdef _WIN32
    hDll = LoadLibraryA(dllPath.c_str());
    if (!hDll) {
        DWORD errorCode = GetLastError();
        LPVOID errorMsg;

        FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, errorCode, 0, (LPSTR)&errorMsg, 0, NULL
            );

        std::ostringstream oss;
        oss << "Failed to load LlamaEngine.dll! Error code: " << errorCode << " - " << (char*)errorMsg;

        LocalFree(errorMsg); // Free allocated memory

        throw std::runtime_error(oss.str());
    }

    loadModelFunc = (LoadModelFunc)GetProcAddress(hDll, "loadModel");
    generateResponseFunc = (GenerateResponseFunc)GetProcAddress(hDll, "generateResponse");
    parseGGUFFunc = (ParseGGUFFunc)GetProcAddress(hDll, "parseGGUF");

    if (!loadModelFunc || !generateResponseFunc || !parseGGUFFunc) {
        FreeLibrary(hDll);
        throw std::runtime_error("Failed to locate functions in LlamaEngine.dll!");
    }

#elif __APPLE__
    hDll = dlopen(dllPath.c_str(), RTLD_LAZY);
    if (!hDll) {
        const char* errorMsg = dlerror();
        std::ostringstream oss;
        oss << "Failed to load LlamaEngine.dylib! Error: " << (errorMsg ? errorMsg : "Unknown error");
        throw std::runtime_error(oss.str());
    }

    loadModelFunc = (LoadModelFunc)dlsym(hDll, "loadModel");
    generateResponseFunc = (GenerateResponseFunc)dlsym(hDll, "generateResponse");
    parseGGUFFunc = (ParseGGUFFunc)dlsym(hDll, "parseGGUF");

    if (!loadModelFunc || !generateResponseFunc || !parseGGUFFunc ) {
        const char* errorMsg = dlerror();
        std::ostringstream oss;
        oss << "Failed to locate functions in LlamaEngine.dylib! Error: " << (errorMsg ? errorMsg : "Unknown error");
        dlclose(hDll);
        throw std::runtime_error(oss.str());
    }
#endif
}

// Destructor: Unloads the DLL
LlamaClient::~LlamaClient() {
#ifdef _WIN32
    if (hDll) {
        FreeLibrary(hDll);
    }
#elif __APPLE__
    if (hDll) {
        dlclose(hDll);
    }
#endif
}


LlamaClient* LlamaClient::Create(const std::string &backendType, const std::string& dllPath) {
    createError.clear();
    try {
        return new LlamaClient(backendType, dllPath);
    } catch (const std::exception& e) {
        createError = e.what();
        return nullptr;
    }
}

const std::string& LlamaClient::GetCreateError() {
    return createError;
}

// Load model
bool LlamaClient::loadModel(const std::string& backendType, struct ModelParameter* params, size_t paramCount, void (*callback)(const char*)) {
    return loadModelFunc(backendType.c_str(), params, paramCount, callback);
}

// Generate response with callback


//std::string LlamaClient::generateResponse(const std::string& prompt, void (*streamCallback)(const char*), void (*finishedCallback)(const char*)) {
bool LlamaClient::generateResponse(const std::string& prompt,
                                   void (*streamCallback)(const char* msg, void* user_data),
                                   void (*finishedCallback)(const char* msg, void* user_data),
                                   void *userData)
{

    // Wrapper function to call the std::function stored in user_data
    auto callbackWrapper = [](const char* msg, void* user_data) {
        auto* func = static_cast<std::function<void(const char*)>*>(user_data);
        if (func) {
            (*func)(msg);  // Invoke the callback
        }
    };

    // Passer un pointeur au lieu de caster directement
    //void* userData = static_cast<void*>(&callbackWrapper);

    bool resultStatus = generateResponseFunc(prompt.c_str(), streamCallback, userData);

    //if(finishedCallback)
    //    finishedCallback(result);

    //return std::string(result);
    return resultStatus;
}

// Parse GGUF file
GGUFMetadata LlamaClient::parseGGUF(const std::string& filepath, void (*callback)(const char*message)) {

    GGUFMetadata metadata;

    //(*GGUFAttributeCallback)(const char* key, GGUFType type, void* value)
    // Call to parseGGUF with a lambda callback to populate metadata

    // User data structure to hold metadata and callback function
    struct UserData {
        GGUFMetadata* metadata;
        void (*callback)(const char* message);
    };

    // Create the user data structure
    UserData userData = { &metadata, callback };

    // Call to parseGGUF with a lambda callback to populate metadata
    parseGGUFFunc(filepath.c_str(),
      [](const char* key, GGUFType type, void* data, void *userData)
    {
          // Cast userData to UserData* and extract metadata and callback
          UserData* dataPtr = static_cast<UserData*>(userData);
          GGUFMetadata* metadataPtr = dataPtr->metadata;
          void (*callback)(const char*) = dataPtr->callback;

          if (!metadataPtr)
              return;

          // Process the metadata entry based on its type
          if (type == TYPE_UINT32) {
              metadataPtr->entries[key] = GGUFMetadataEntry(*static_cast<uint32_t*>(data));
          } else if (type == TYPE_STRING) {
              metadataPtr->entries[key] = GGUFMetadataEntry(static_cast<const char*>(data));
          } else {
              metadataPtr->entries[key] = GGUFMetadataEntry("[Unknown Type]");
          }

          // Invoke the callback with the message
          if (callback) {
              std::string message = key;
              message += ": " + metadataPtr->entries[key].toString(); // Use toString() here
              callback(message.c_str());  // Call the callback with the message
          }

    }, callback, &userData);  // Pass the userData structure

    // Convert model name to GGUFMetadata entry if available
    if (!metadata.entries["model_name"].svalue.empty()) {
        // Assuming metadata has model_name entry processed by the callback
        metadata.entries["model_name"] = GGUFMetadataEntry(metadata.entries["model_name"].svalue);
    }

    return metadata;
}

std::string LlamaClient::backendType()
{
    return backend;
}

std::string LlamaClient::libraryName()
{
    return library;
}
