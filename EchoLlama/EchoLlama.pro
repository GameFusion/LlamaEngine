# -------------------------------------------------
# EchoLlama.pro - QMake Project File for EchoLlama
# -------------------------------------------------

QT += core gui widgets network
CONFIG += c++20
# Target Configuration
TEMPLATE = app
RESOURCES += resources.qrc

# Source Files
SOURCES += \
    main.cpp \
    EchoLlama.cpp \
    FontAwesome.cpp \
    DownloadManager.cpp \
    NetworkUtils.cpp \
    ../LlamaClient.cpp

HEADERS += \
    EchoLlama.h \
    FontAwesome.h \
    DownloadManager.h \
    NetworkUtils.h \
    ResponseWorker.h \
    ..//LlamaClient.h

# Include Paths
INCLUDEPATH += \
    .\
    ../

# Output Directory
DESTDIR = .

# Export Macros
DEFINES += \
    QT_NO_DEBUG

CONFIG(debug, debug|release) {
    DEFINES += DEBUG
}

# Configuration for different platforms
win32: {
    DEFINES += WIN32
}
macx: {
    DEFINES += MACX
    QMAKE_CFLAGS += -include arm_acle.h
    QMAKE_CXXFLAGS += -include arm_acle.h
    QMAKE_LFLAGS += -Wl,-rpath,$$PWD/../build/Qt_6_10_2_for_macOS-Debug/bin/Metal
    QMAKE_LFLAGS += -Wl,-rpath,/opt/local/lib
    LLAMA_RUNTIME_VERSION = gguf-v0.4.0-3652-gef19c717
    LLAMA_RUNTIME_DIR = "$$OUT_PWD/EchoLlama.app/Contents/Resources/llama.cpp/$$LLAMA_RUNTIME_VERSION/Metal"
    QMAKE_POST_LINK += mkdir -p $$LLAMA_RUNTIME_DIR && cp -P "$$PWD/../build/Qt_6_10_2_for_macOS-Debug/bin/Metal"/libLlamaEngine*.dylib $$LLAMA_RUNTIME_DIR/ && cp /opt/local/lib/libllama.dylib /opt/local/lib/libggml*.dylib $$LLAMA_RUNTIME_DIR/
}
unix: {
    DEFINES += UNIX
}

# QMake Configuration
CONFIG += c++17

# Additional Options
QMAKE_CXXFLAGS += -Wall

# Final Project File
TARGET = EchoLlama
