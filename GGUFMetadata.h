#ifndef GGUF_METADATA_H
#define GGUF_METADATA_H

#include <string>
#include <unordered_map>

// GGUF Type Enum
enum GGUFType {
    TYPE_UNKNOWN,
    TYPE_UINT32,
    TYPE_STRING
};

// GGUF Metadata Entry
struct GGUFMetadataEntry {
    GGUFType type;
    uint32_t ivalue;
    std::string svalue;

    GGUFMetadataEntry() : type(TYPE_UNKNOWN), ivalue(0) {}
    GGUFMetadataEntry(uint32_t value) : type(TYPE_UINT32), ivalue(value) {}
    GGUFMetadataEntry(const std::string& value) : type(TYPE_STRING), svalue(value) {}

    std::string toString() const {
        switch (type) {
        case TYPE_UINT32:
            return std::to_string(ivalue);
        case TYPE_STRING:
            return svalue;
        default:
            return "[Unknown Type]";
        }
    }
};

// GGUF Metadata Container
class GGUFMetadata {
public:
    std::unordered_map<std::string, GGUFMetadataEntry> entries;

    // Get the maximum context length
    int getMaxContextLength() const {
        for (const auto& entry : entries) {
            if (entry.first.find("context_length") != std::string::npos) {
                return entry.second.ivalue;
            }
        }
        return -1;
    }
};

#endif // GGUF_METADATA_H
