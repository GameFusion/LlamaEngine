
// these values are pulled from llama.cpp :
//    ../..//llama.cpp-v1/build/llama-config.cmake
const char* LLAMA_VERSION="0.0.4658";
const char* LLAMA_BUILD_COMMIT="855cd073";

const char* LLAMA_COMMIT="855cd073";
 int LLAMA_BUILD_NUMBER=4658;
const char* LLAMA_BUILD_TARGET="macOS";
const char* LLAMA_COMPILER="clang";
const bool LLAMA_SHARED_LIB=0;

#include <stdio.h>



#include <cstdarg>
#include <cstdio>
// Define the log verbosity threshold as a global variable
#include "common.h"
//#include "log.h"

//ggml_log_level _common_log_verbosity_thold = GGML_LOG_LEVEL_INFO; // Default to INFO
/*
struct common_log;

void common_log_add(common_log* log, ggml_log_level level, const char* format, ...) {
    if (!log) return;

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, "\n");  // Ensure newline
}
void common_log_main() {
    // Initialize logging system if needed
    printf("Common log system initialized.\n");
}
*/
