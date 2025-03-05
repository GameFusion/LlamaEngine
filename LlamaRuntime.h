#ifndef LLAMARUNTIME_H
#define LLAMARUNTIME_H

#include <string>
#include <vector>
#include <functional>
#include <iostream> // Optional: fallback to console output
#include "llama.h"
#include "gguf.h"

#include "GGUFMetadata.h"

class LlamaRuntime {

public:
    LlamaRuntime();
    ~LlamaRuntime();

    bool loadModel();
    bool loadModelInternal(const std::string &modelPath, int ngl, int n_ctx);

    void setModelPath(const std::string &path);
    void setContextSize(int size);
    void setTemperature(float temp);
    void setTopK(float k);
    void setTopP(float p);
    void setRepetitionPenalty(float penalty);

    bool generateResponse(const std::string &input_prompt, void (*callback)(const char*, void *userData), void *userData);

    static GGUFMetadata parseGGUF(const std::string& filepath, void(*callback)(const char* message));

    // Logging and logginc callbacks
    using LogCallback = std::function<void(const std::string&)>;

    void setLogCallback(LogCallback callback);

    void logMessage(const std::string& message);
    // help function that format message type like '[ERROR] message'
    void logInfo(const std::string& errorMessage);
    void logError(const std::string& errorMessage);
    void logWarning(const std::string& warningMessage);

private:
    bool generate(const std::string &prompt, void (*callback)(const char*, void *), void *userData);
    std::vector<llama_token> tokenizePrompt(const std::string &prompt, bool is_first);

    llama_context *ctx = nullptr;
    llama_model *model = nullptr;
    llama_sampler *smpl = nullptr;
    const llama_vocab *vocab = nullptr;

    // $ git describe
    static std::string llama_version;
    // $ git log - 1 --format = % cd
    static std::string llama_date;

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted;
    std::string error_;
    std::vector<std::string> errorMessages;
    std::string response;

    float temperature = 0.8f;
    int context_size = 4096;
    std::string modelPath;
    float topK = 40;
    float topP = 1.0;
    float repetitionPenalty = 1.0f;

    LogCallback logCallback;
};

#endif // LLAMARUNTIME_H

