TEMPLATE = lib



# Backend Selection (CPU, CUDA, Vulkan, etc.)
isEmpty(BACKEND){
    BACKEND = CUDA # Change this to CUDA or Vulkan when needed
    mac {
    BACKEND = Metal # Change this to CUDA or Vulkan when needed
    }
}

message(BACKEND set to $$(BACKEND))

# Set Target Name Based on Backend
TARGET = LlamaEngine

# Set Backend-Specific Output Directory
DESTDIR = bin/$${BACKEND}

GF=$$(GameFusion)

isEmpty(GF) {
    GF=../..
    message(Not found found GameFusion setting value to $$GF)
} else {
    message(Found GameFusion at $$(GameFusion))
    GF=$$(GameFusion)
}

mac {
    GF=/Users/andreascarlen/GameFusion
}

QT -= gui qt core QtCore

DEFINES += LlamaEngine_EXPORTS

INCLUDEPATH += $$PWD/include

SOURCES += LlamaEngine.cpp LlamaRuntime.cpp LlamaRuntimeVision.cpp
HEADERS += LlamaEngine.h LlamaRuntime.h LlamaRuntimeVision.h
HEADERS += LlamaSession.h PromptResponse.h

SOURCES += llama_version.cpp


# macOS-specific settings
mac {

    GF=/Users/andreascarlen/GameFusion
    CONFIG -= app_bundle # Ensure it's not treated as a macOS application bundle
    CONFIG += dylib # Use dylib instead of dll on macOS



    LIBS += -L/opt/local/lib
    # Common Libraries
    LIBS += -lllama
    LIBS += -lggml
    LIBS += -lggml-base
    LIBS += -lggml-cpu
    LIBS += -lggml-metal

    INCLUDEPATH += /opt/local/include
    INCLUDEPATH += ../../../../ExternalCode/llama.cpp/examples/llava
    INCLUDEPATH += ../../../../ExternalCode/llama.cpp/common
    SOURCES += ../../../../ExternalCode/llama.cpp/examples/llava/clip.cpp
    SOURCES += ../../../../ExternalCode/llama.cpp/common/common.cpp
    SOURCES += ../../../../ExternalCode/llama.cpp/common/sampling.cpp
    SOURCES += ../../../../ExternalCode/llama.cpp/common/console.cpp
    SOURCES += ../../../../ExternalCode/llama.cpp/common/log.cpp
    HEADERS += llama_version.h

    # Set the macOS library name prefix to an empty string
    #QMAKE_LIB_PREFIX =  # This removes the default 'lib' prefix on macOS

    # Explicitly set the output name to LlamaEngine.dylib
    #TARGET = LlamaEngine.dylib  # Ensure the target name includes the .dylib extension


    # Optional: Set the output name explicitly with .dylib extension
    DESTDIR = bin/$${BACKEND}
}

# Windows-specific settings
win32: {

    CONFIG += dll

    CONFIG(debug, debug|release) {
        TARGET = $$join(TARGET,,,d)
    }

    # Generate import library and DLL
    QMAKE_LFLAGS += /DLL
    QMAKE_LFLAGS_RELEASE += /OPT:REF
    QMAKE_LFLAGS_DEBUG += /DEBUG

    # Backend-Specific Includes and Libraries
    contains(BACKEND, CPU) {
        INCLUDEPATH += $$GF/Programmes/llama.cpp/include
        LIBS += -L$$GF/Programmes/llama.cpp/lib
        VCPROJ_NAME = LlamaEngineCPU

        message(Backend-Specific  CPU)
    }
    contains(BACKEND, CUDA) {
        INCLUDEPATH += $$GF/Programmes/llama.cpp/cuda/include
        LIBS += -L$$GF/Programmes/llama.cpp/cuda/lib
        LIBS += -lggml-cuda
        VCPROJ_NAME = LlamaEngineCUDA
         message(Backend-Specific  CUDA)
    }
    contains(BACKEND, Vulkan) {
        INCLUDEPATH += $$GF/Programmes/llama.cpp/vulkan/include
        LIBS += -L$$GF/Programmes/llama.cpp/vulkan/lib
        LIBS += -lggml-vulkan
        VCPROJ_NAME = LlamaEngineVulkan
         message(Backend-Specific  Vulkan)
    }

    # Common Libraries
    LIBS += -lllama
    LIBS += -lggml
    LIBS += -lggml-base
    LIBS += -lggml-cpu

    # Set the Visual Studio project name
    win32:QMAKE_PROJECT_NAME = $$VCPROJ_NAME
}
