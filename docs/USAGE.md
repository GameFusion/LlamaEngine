# LlamaClient Usage Example

Below is a full example illustrating how to use LlamaClient to load a model and generate responses.

## Full Example

### 1. Load the Model

Before generating a response, you'll need to load the model into memory.

```cpp
#include "llamaClient.h"
#include <iostream>

int main() {
    // Create the LlamaClient
    LlamaClient* client = LlamaClient::Create("CUDA", "LlamaEngine.dll");

    if (!client) {
        std::cerr << "Failed to initialize LlamaClient!" << std::endl;
        return 1;
    }

    // Model loading parameters
    float temperature = 0.7f;
    int contextSize = 4096;  // Set context size
    float topK = 40.0f;
    float topP = 0.9f;
    float repetitionPenalty = 1.2f;
    std::string modelPath = "path/to/your/model";  // Replace with your model path

    struct ModelParameter params[] = {
        {"temperature", PARAM_FLOAT, &temperature},
        {"context_size", PARAM_INT, &contextSize},
        {"top_k", PARAM_FLOAT, &topK},
        {"top_P", PARAM_FLOAT, &topP},
        {"repetition_penalty", PARAM_FLOAT, &repetitionPenalty},
        {"model_path", PARAM_STRING, (void*)modelPath.c_str()}
    };

    size_t paramCount = sizeof(params) / sizeof(params[0]);

    if (!client->loadModel(modelPath, params, paramCount, [](const char* message) {
        std::cout << "Model loading: " << message << std::endl;
    })) {
        std::cerr << "Failed to load model!" << std::endl;
        delete client;
        return 1;
    }

    // Generate response using lambdas
    client->generateResponse("Hello, Llama!", 
        [](const char* msg, void* userData) {
            std::cout << msg;  // Streamed tokens
        }, 
        [](const char* msg, void* userData) {
            std::cout << "\nFinal Response: " << msg << std::endl;  // Final response
        }, 
        nullptr);

    // Cleanup
    delete client;
    return 0;
}
```