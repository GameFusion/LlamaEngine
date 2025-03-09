#ifndef LlamaSession_h
#define LlamaSession_h

#include <list>
#include <string>
#include <ctime>
#include <uuid/uuid.h>

#include "llama.h"
#include "gguf.h"

#include "PromptResponse.h"

/**
 * @brief Represents an interactive session with the Llama model.
 *
 * This class manages session history, model context, and sampling configurations.
 * Each session is uniquely identified and maintains a conversation history,
 * allowing for contextual responses across multiple interactions.
 */
class LlamaSession {
public:
    std::list<PromptResponse> history; ///< Stores the prompt-response history.
    std::string contextBuffer; ///< Cached session context for model interactions.
    std::string tag; ///< Optional session tag for categorization.
    std::string sessionName; ///< Human-readable session name.
    std::string sessionId; ///< Unique session identifier.
    time_t timestamp; ///< Creation time of the session.
    llama_context* ctx = nullptr; ///< Pointer to the model's runtime context.
    llama_sampler *smpl = nullptr; ///< Pointer to the sampling handler.

    /**
     * @brief Creates a new LlamaSession with a unique session ID.
     *
     * @param name Human-readable session name.
     * @param context Pointer to the Llama model context.
     * @param sampler Pointer to the Llama sampler.
     */
    LlamaSession(std::string name, llama_context* context, llama_sampler *sampler)
        : sessionName(std::move(name)), timestamp(time(nullptr)), ctx(context), smpl(sampler)
    {
        uuid_t uuid;
        char uuidStr[37];
        uuid_generate(uuid);
        uuid_unparse(uuid, uuidStr);
        sessionId = std::string(uuidStr); ///< Generates a unique session ID.
    }

    /**
     * @brief Destructor that safely frees session resources.
     */
    ~LlamaSession() {
        clearSampler();
        clearContext();
    }

    /**
     * @brief Clears the sampler, freeing associated memory.
     */
    void clearSampler(){
        if (smpl) {
            llama_sampler_free(smpl);
            smpl = nullptr;
        }
    }

    /**
     * @brief Clears the model context, resetting its state.
     */
    void clearContext(){
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
    }

    /**
     * @brief Updates the context buffer using the session history.
     *
     * This function aggregates non-ignored prompt-response pairs
     * into a single string used for contextual model interactions.
     */
    void updateContextBuffer() {
        contextBuffer.clear();
        for (const auto& entry : history) {
            if (entry.flag != PromptResponse::Flag::IGNORE) {
                contextBuffer += entry.prompt + " " + entry.response + " ";
            }
        }
        timestamp = time(nullptr); // Update session timestamp
    }
};

#endif // LlamaSession_h
