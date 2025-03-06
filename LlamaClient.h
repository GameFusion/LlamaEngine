#ifndef LLAMA_CLIENT_H
#define LLAMA_CLIENT_H

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

/*
// Include the same structures from LlamaEngine
struct GGUFMetadata {
    std::string name;
    std::vector<std::string> attributes;
};

struct Attachment {
    std::string filename;
    std::vector<char> data;
};
*/

// LlamaClient Class
class LlamaClient {
public:
#ifdef _WIN32
    LlamaClient(const std::string &backend = "CUDA", const std::string& dllPath = "LlamaEngine.dll");
#elif __APPLE__
    LlamaClient(const std::string &backend = "CPU", const std::string& dllPath = "LlamaEngine.dylib");
#endif
    ~LlamaClient();

    std::string backendType();
    std::string libraryName();

    static LlamaClient* Create(const std::string &backend /*= "CUDA"*/, const std::string& dllPath /*= "LlamaEngined.dll"*/);
    static const std::string& GetCreateError();

    bool loadModel(const std::string& modelName, struct ModelParameter* params, size_t paramCount, void (*callback)(const char*) = nullptr);
    //std::string generateResponse(const std::string& prompt, void (*streamCallback)(const char*) = nullptr, void (*finishedCallback)(const char*) = nullptr);
    bool generateResponse(const std::string& prompt,
                          void (*streamCallback)(const char* msg, void* user_data),
                          void (*finishedCallback)(const char* msg, void* user_data), void *userData);

    GGUFMetadata parseGGUF(const std::string& filepath, void (*callback)(const char* message));

private:
#ifdef _WIN32
    HMODULE hDll;
    typedef bool (*LoadModelFunc)(const char*, struct ModelParameter* params, size_t paramCount, void (*)(const char*));
    typedef bool (*GenerateResponseFunc)(const char*, void (*)(const char* msg, void* user_data), void (*)(const char* completeResponse, void* user_data), void *userData);
    typedef const char* (*ParseGGUFFunc)(const char*, void (*)(const char* key, GGUFType type, void* data, void *userData), void (*callback)(const char* message), void *userData);

#elif __APPLE__
    void* hDll;
    typedef bool (*LoadModelFunc)(const char*, struct ModelParameter* params, size_t paramCount, void (*)(const char*));
    typedef bool (*GenerateResponseFunc)(const char*, void (*)(const char* msg, void* user_data), void *userData);
    typedef GGUFMetadata (*ParseGGUFFunc)(const char*, void (*)(const char* key, GGUFType type, void* data, void *userData), void (*callback)(const char* message), void *userData);

#endif

    LoadModelFunc loadModelFunc;
    GenerateResponseFunc generateResponseFunc;
    ParseGGUFFunc parseGGUFFunc;

    void responseCallback(const std::string& response);
    void finishedCallback(const std::string& message);

    static std::string createError; // set from static create method

    std::string backend;   // Store the backend (CPU, CUDA, Vulkan)
    std::string library;
};

#endif // LLAMA_CLIENT_H
