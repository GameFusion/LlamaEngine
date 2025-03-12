#include "LlamaClient.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <dlfcn.h>
#endif

// Static member variable for storing creation error messages
std::string LlamaClient::createError;

bool SetemLoadLibrary(const std::string& relativePath, LlamaClient** clientPtr, const std::string& backendType) {
    // Verify if the file exists
    struct stat buffer;
    if (stat(relativePath.c_str(), &buffer) != 0) {
        std::cerr << "File does not exist: " << relativePath << std::endl;
        return false;
    }

    std::cout << "File exists: " << relativePath << std::endl;

    // Get the absolute directory path and filename
    std::filesystem::path filePath(relativePath);
    std::string libraryPath = std::filesystem::absolute(filePath.parent_path()).string();
    std::string fileName = filePath.filename().string();

    std::cout << "Library Path: " << libraryPath << std::endl;

#ifdef WIN32
    // Convert to wide string for Windows API
    std::wstring wLibraryPath(libraryPath.begin(), libraryPath.end());

    // Set the DLL directory using the absolute path
    if (SetDllDirectoryW(wLibraryPath.c_str())) {
        std::cout << "SetDllDirectoryW succeeded: " << libraryPath << std::endl;

        // Now initialize the LlamaClient with just the filename, not the full path
        try {
            *clientPtr = new LlamaClient(backendType, fileName);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize LlamaClient: " << e.what() << std::endl;
            return false;
        }
    } else {
        DWORD error = GetLastError();
        std::cerr << "SetDllDirectoryW failed! Error code: " << error << std::endl;
        return false;
    }
#else
    // For non-Windows platforms, use the full path
    try {
        *clientPtr = new LlamaClient(backendType, relativePath);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize LlamaClient: " << e.what() << std::endl;
        return false;
    }
#endif
}
/**
 * @brief Constructor for LlamaClient.
 * @param backendType The type of backend used.
 * @param dllPath The path to the dynamic library (DLL/shared object).
 * @throws std::runtime_error if the library fails to load or required functions are not found.
 */
LlamaClient::LlamaClient(const std::string &backendType, const std::string& dllPath) {
    backend = backendType;
    library = dllPath;

    LoadLibrary(dllPath);
}

void LlamaClient::LoadLibrary(const std::string& dllPath)
{
    std::string relativePath = dllPath;

    // Verify if the file exists
    struct stat buffer;
    if (stat(relativePath.c_str(), &buffer) != 0) {
        std::cerr << "File does not exist: " << relativePath << std::endl;
        std::ostringstream oss;
        oss << "File does not exist: " << relativePath;
        createError = oss.str();
        throw std::runtime_error(oss.str());
    }

    std::cout << "File exists: " << relativePath << std::endl;

    // Get the absolute path
    char absolutePath[MAX_PATH];
    GetFullPathNameA(relativePath.c_str(), MAX_PATH, absolutePath, NULL);
    std::string fullPath = absolutePath;

    // Extract directory path - find last backslash
    size_t lastSlash = fullPath.find_last_of("\\/");
    std::string libraryPath;
    std::string fileName;

    if (lastSlash != std::string::npos) {
        fileName = fullPath.substr(lastSlash + 1);
        libraryPath = fullPath.substr(0, lastSlash);
    } else {
        // No path separator found - use current directory
        fileName = relativePath;
        char currentDir[MAX_PATH];
        GetCurrentDirectoryA(MAX_PATH, currentDir);
        libraryPath = currentDir;
    }

    std::cout << "Library Path: " << libraryPath << std::endl;
    std::cout << "File Name: " << fileName << std::endl;

#ifdef _WIN32

    // Convert to wide string for Windows API
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, libraryPath.c_str(), -1, NULL, 0);
    wchar_t* wLibraryPath = new wchar_t[size_needed];
    MultiByteToWideChar(CP_UTF8, 0, libraryPath.c_str(), -1, wLibraryPath, size_needed);

    // Set the DLL directory using the absolute path
    bool success = false;
    if (SetDllDirectoryW(wLibraryPath)) {

        std::cout << "SetDllDirectoryW succeeded: " << libraryPath << std::endl;

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

        // Load function pointers
        loadModelFunc = (LoadModelFunc)GetProcAddress(hDll, "loadModel");
        generateResponseFunc = (GenerateResponseFunc)GetProcAddress(hDll, "generateResponse");
        parseGGUFFunc = (ParseGGUFFunc)GetProcAddress(hDll, "parseGGUF");
        getContextInfoFunc = (GetContextInfoFunc)GetProcAddress(hDll, "getContextInfo");

        if (!loadModelFunc || !generateResponseFunc || !parseGGUFFunc || !getContextInfoFunc) {
            FreeLibrary(hDll);
            throw std::runtime_error("Failed to locate functions in LlamaEngine.dll!");
        }
    } else {
        DWORD error = GetLastError();
        std::cerr << "SetDllDirectoryW failed! Error code: " << error << std::endl;
    }

    // Clean up allocated memory
    delete[] wLibraryPath;

#elif __APPLE__
    hDll = dlopen(dllPath.c_str(), RTLD_LAZY);
    if (!hDll) {
        const char* errorMsg = dlerror();
        std::ostringstream oss;
        oss << "Failed to load LlamaEngine.dylib! Error: " << (errorMsg ? errorMsg : "Unknown error");
        createError = oss.str();
        throw std::runtime_error(oss.str());
    }

    loadModelFunc = (LoadModelFunc)dlsym(hDll, "loadModel");
    generateResponseFunc = (GenerateResponseFunc)dlsym(hDll, "generateResponse");
    parseGGUFFunc = (ParseGGUFFunc)dlsym(hDll, "parseGGUF");
    getContextInfoFunc = (GetContextInfoFunc)dlsym(hDll, "getContextInfo");

    if (!loadModelFunc || !generateResponseFunc || !parseGGUFFunc || !getContextInfoFunc) {
        const char* errorMsg = dlerror();
        std::ostringstream oss;
        oss << "Failed to locate functions in LlamaEngine.dylib! Error: " << (errorMsg ? errorMsg : "Unknown error");
        dlclose(hDll);
        createError = oss.str();
        throw std::runtime_error(oss.str());
    }
#endif
}

/**
 * @brief Destructor for LlamaClient. Unloads the DLL.
 */
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

/**
 * @brief Factory method to create a LlamaClient instance.
 * @param backendType The backend type to use.
 * @param dllPath The path to the dynamic library.
 * @return A pointer to LlamaClient or nullptr on failure.
 */
LlamaClient* LlamaClient::Create(const std::string &backendType, const std::string& dllPath) {
    createError.clear();
    try {
        return new LlamaClient(backendType, dllPath);
    } catch (const std::exception& e) {
        createError = e.what();
        return nullptr;
    }
}

/**
 * @brief Retrieves the last creation error message.
 * @return A reference to the error message string.
 */
const std::string& LlamaClient::GetCreateError() {
    return createError;
}

/**
 * @brief Loads the model.
 * @param backendType The type of backend used.
 * @param params Model parameters.
 * @param paramCount Number of parameters.
 * @param callback Callback function.
 * @return True if successful, false otherwise.
 */
bool LlamaClient::loadModel(const std::string& backendType, struct ModelParameter* params, size_t paramCount, void (*callback)(const char*)) {
    return loadModelFunc(backendType.c_str(), params, paramCount, callback);
}

/**
 * @brief Generates a response from the model.
 * @param prompt The input prompt.
 * @param streamCallback Streaming callback.
 * @param finishedCallback Finished response callback.
 * @param userData User data pointer.
 * @return True if the response was generated successfully, false otherwise.
 */
bool LlamaClient::generateResponse(const std::string& prompt,
                                   void (*streamCallback)(const char* msg, void* user_data),
                                   void (*finishedCallback)(const char* msg, void* user_data),
                                   void *userData)
{
    const int sessionId = 0;
    return generateResponseFunc(sessionId, prompt.c_str(), streamCallback, finishedCallback, userData);
}

/**
 * @brief Generates a response from the model.
 * @param sessionId The unique identifier for the session.
 * @param prompt The input prompt.
 * @param streamCallback Streaming callback.
 * @param finishedCallback Finished response callback.
 * @param userData User data pointer.
 * @return True if the response was generated successfully, false otherwise.
 */
bool LlamaClient::generateResponse(int sessionId,
                                   const std::string& prompt,
                                   void (*streamCallback)(const char* msg, void* user_data),
                                   void (*finishedCallback)(const char* msg, void* user_data),
                                   void *userData)
{
    return generateResponseFunc(sessionId, prompt.c_str(), streamCallback, finishedCallback, userData);
}

/**
 * @brief Parses a GGUF file and extracts metadata.
 * @param filepath Path to the GGUF file.
 * @param callback Callback function.
 * @return A GGUFMetadata object containing extracted metadata.
 */
GGUFMetadata LlamaClient::parseGGUF(const std::string& filepath, void (*callback)(const char*message)) {

    GGUFMetadata metadata;

    // User data structure to hold metadata and callback function
    struct UserData {
        GGUFMetadata* metadata;
        void (*callback)(const char* message);
    };

    // Create the userData structure to pass data to lambda
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

/**
 * @brief Gets the backend type used by the client.
 * @return The backend type as a string.
 */
std::string LlamaClient::backendType()
{
    return backend;
}

/**
 * @brief Gets the library name used by the client.
 * @return The library name as a string.
 */
std::string LlamaClient::libraryName()
{
    return library;
}

std::string LlamaClient::getContextInfo(){

    std::string result;
    getContextInfoFunc([](const char *info, void *userData){
        // Cast userData to std::string reference
        std::string &result = *static_cast<std::string*>(userData);

        // Assign the result to the string passed through userData
        result = info;
    }, &result);

    return result;
}
