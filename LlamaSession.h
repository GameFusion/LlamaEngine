#ifndef LlamaSession_h
#define LlamaSession_h

#include <list>
#include <string>
#include <ctime>

#ifdef WIN32

#include <intrin.h>

/**
 * @brief Represents a UUID structure for session identification.
 *
 * This structure generates and stores a unique identifier using the Time Stamp Counter (TSC).
 */
struct UUID {
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];

    /**
     * @brief Generates a random UUID using the TSC.
     */
    void generate() {
        Data1 = __rdtsc(); // Use TSC for entropy
        Data2 = (unsigned short)(__rdtsc() >> 16);
        Data3 = (unsigned short)(__rdtsc() & 0xFFFF);

        for (int i = 0; i < 8; ++i) {
            Data4[i] = (unsigned char)(__rdtsc() >> (i * 4));
        }

        // Set UUID version to 4 (random-based)
        Data3 = (Data3 & 0x0FFF) | (4 << 12);
        // Set variant (RFC 4122)
        Data4[0] = (Data4[0] & 0x3F) | 0x80;
    }

    /**
     * @brief Converts the UUID to a string representation.
     * @return A string in the format "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
     */
    std::string toString() const {
        char buffer[37];  // UUID string is 36 characters + null terminator
        std::snprintf(buffer, sizeof(buffer),
                      "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                      Data1, Data2, Data3,
                      Data4[0], Data4[1],
                      Data4[2], Data4[3], Data4[4], Data4[5], Data4[6], Data4[7]);

        return std::string(buffer);
    }
};


#else
#include <uuid/uuid.h>
#endif

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
#ifdef SESSION_TEST
    std::list<PromptResponse> history; ///< Stores the prompt-response history.
    std::string contextBuffer; ///< Cached session context for model interactions.
    std::string tag; ///< Optional session tag for categorization.
#endif
    std::string sessionName; ///< Human-readable session name.
    std::string sessionId; ///< Unique session identifier.

    time_t timestamp; ///< Creation time of the session.
    llama_context* ctx = nullptr; ///< Pointer to the model's runtime context.
    llama_sampler *smpl = nullptr; ///< Pointer to the sampling handler.

    std::vector<llama_chat_message> messages; ///< Stores chat messages.
    std::vector<char> formatted;              ///< Formatted message buffer.
    std::string response;                     ///< Last generated response.

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

#ifdef WIN32
        UUID uuid;
        uuid.generate();
        sessionId = uuid.toString();
#else
        uuid_t uuid;
        char uuidStr[37];
        uuid_generate(uuid);
        uuid_unparse(uuid, uuidStr);
        sessionId = std::string(uuidStr); ///< Generates a unique session ID.
#endif
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
     * @brief Clears the session history and chat messages.
     *
     * This function frees allocated message content and clears the KV cache.
     */
    void clearHistory(){
        // Free allocated message content
        for (auto &msg : messages) {
            free(const_cast<char *>(msg.content));
        }

        messages.clear();

        //Explicitly Clear the KV Cache
        llama_kv_cache_clear(ctx);
    }

    /**
     * @brief Updates the context buffer using the session history.
     *
     * This function aggregates non-ignored prompt-response pairs
     * into a single string used for contextual model interactions.
     */
    void updateContextBuffer() {
#ifdef SESSION_TEST
        contextBuffer.clear();
        for (const auto& entry : history) {
            if (entry.flag != PromptResponse::Flag::IGNORE) {
                contextBuffer += entry.prompt + " " + entry.response + " ";
            }
        }
#endif
        timestamp = time(nullptr); // Update session timestamp
    }
};

#endif // LlamaSession_h
