#ifndef LlamaEngine_h
#define LlamaEngine_h

#include "GGUFMetadata.h"

// Define the export/import macro based on the platform
#ifdef _WIN32
#ifdef LlamaEngine_EXPORTS
#define LlamaEngine_API __declspec(dllexport)
#else
#define LlamaEngine_API __declspec(dllimport)
#endif
#elif __APPLE__
#ifdef LlamaEngine_EXPORTS
#define LlamaEngine_API __attribute__((visibility("default")))
#else
#define LlamaEngine_API
#endif
#else
// For other platforms, you can define a default behavior or leave it empty
#define LlamaEngine_API
#endif

// C-compatible structures
struct LlmMetadata {
    const char* name;
    const char** attributes;  // array of C-strings
    size_t attribute_count;   // number of attributes
};


// Define the parameter struct
typedef enum {
    PARAM_FLOAT,
    PARAM_INT,
    PARAM_STRING,
    PARAM_UNKNOWN
} ParamType;

struct ModelParameter {
    const char* key;        // The name of the parameter (e.g., "temperature")
    ParamType type;         // Type of the parameter (e.g., float, int, etc.)
    void* value;            // Pointer to the parameter's value
};

extern "C" {
    // Exported functions using C-compatible types

    // Updated loadModel function with parameter array
    LlamaEngine_API bool loadModel(const char* backendType,
                                   struct ModelParameter* params, size_t paramCount,
                                   void (*callback)(const char*) = nullptr);

    LlamaEngine_API bool generateResponse(const char* prompt, void (*callback)(const char*, void *userData), void *userData);

    // Returns a GGUFMetadata struct
    typedef void (*GGUFAttributeCallback)(const char* key, GGUFType type, void* value, void *user_data);
    LlamaEngine_API char* parseGGUF(const char* filepath, GGUFAttributeCallback callback, void (*messageCallback)(const char* message), void *user_data = nullptr);
}

#endif // LlamaEngine_h
