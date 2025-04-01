
//int main_vision(const char *prompt, const char *image);

//bool hasVision();
//bool generateVision(int session_id, const std::string &input_prompt, void (*callback)(const char*, void *userData), void *userData);

#ifndef LlamaRuntimeVision_h
#define LlamaRuntimeVision_h

#include <string>
#include <vector>

// Forward declarations
struct common_params;
struct gemma3_context;
struct common_sampler;
struct clip_ctx;

/**
 * @class LlamaRuntimeVision
 * @brief A minimalistic wrapper for vision-based LLM functionality.
 *
 * This class encapsulates the existing vision-specific functionality from LlamaRuntimeVision.cpp
 * with minimal changes to the original implementation.
 */
class LlamaRuntimeVision {
public:
    /**
     * @brief Constructor initializes vision runtime
     */
    LlamaRuntimeVision();

    /**
     * @brief Destructor cleans up resources
     */
    ~LlamaRuntimeVision();

    /**
     * @brief Initialize the vision system with model paths and parameters
     *
     * @param model_path Path to the model file
     * @param mmproj_path Path to the multimodal projector file
     * @param temperature Temperature parameter for sampling (default: 0.2)
     * @param context_size Context size in tokens (default: 8192)
     * @return True if initialization was successful
     */
    bool initialize(const std::string& model_path,
                    const std::string& mmproj_path,
                    float temperature = 0.2f,
                    int context_size = 8192);

    /**
     * @brief Check if vision capability is available
     *
     * @return True if vision is available
     */
    bool hasVision();

    /**
     * @brief Process an image and generate a response
     *
     * @param prompt Text prompt to accompany the image
     * @param image_path Path to the image file
     * @param callback Function to receive generated tokens
     * @param userData User data passed to callback
     * @return True if successful
     */
    bool processImageAndGenerate(const std::string& prompt,
                                 const std::string& image_path,
                                 void (*callback)(const char*, void* userData) = nullptr,
                                 void* userData = nullptr);

    /**
     * @brief Generate a response for a prompt (with context from previous images)
     *
     * @param prompt Text prompt
     * @param callback Function to receive generated tokens
     * @param userData User data passed to callback
     * @return True if successful
     */
    bool generateResponse(const std::string& prompt,
                          void (*callback)(const char*, void* userData) = nullptr,
                          void* userData = nullptr);

    /**
     * @brief Clear the context but keep loaded models
     *
     * @param keepImages If true, try to keep images in context
     * @return True if successful
     */
    bool clearContext(bool keepImages = true);

    /**
     * @brief Get info about current context usage
     *
     * @return String containing context usage information
     */
    std::string getContextInfo();

private:
    // The original global variables, now encapsulated as member variables
    common_params* params = nullptr;
    gemma3_context* ctx = nullptr;
    common_sampler* smpl = nullptr;
    int image_end_pos = 0;
    bool is_initialized = false;
};

#endif // LlamaRuntimeVision_h
