#ifndef PromptResponse_h
#define PromptResponse_h

#include <string>
#include <ctime>

/**
 * @brief Represents a single prompt-response pair within a session.
 *
 * This class stores user input (prompt) and the corresponding model-generated
 * response. It also includes metadata such as relevance weighting, timestamp,
 * and a flag to determine how the response should be handled in future interactions.
 */
class PromptResponse {
public:
    std::string prompt;   ///< User input prompt.
    std::string response; ///< Generated response.
    
    /**
     * @brief Flags to control how the response is treated in context.
     */
    enum class Flag {
        IGNORE,        ///< User decides to exclude from future context
        IMPORTANT,     ///< Prioritized for future responses
        INCLUDE        ///< Default: part of the session context
    };

    Flag flag = Flag::INCLUDE; ///< Default behavior: include in session context.
    float relevanceWeight = 1.0f; ///< User-defined weight (higher = more relevant)
    time_t timestamp; ///< Timestamp indicating when the response was generated.

    /**
     * @brief Constructs a PromptResponse object.
     *
     * @param p The input prompt.
     * @param r The generated response.
     * @param f The flag indicating how the response is treated in context.
     * @param w The relevance weight for prioritization.
     */
    PromptResponse(std::string p, std::string r, Flag f = Flag::INCLUDE, float w = 1.0f)
        : prompt(std::move(p)), response(std::move(r)), flag(f), relevanceWeight(w), timestamp(time(nullptr)) {}
};

#endif // PromptResponse_h
