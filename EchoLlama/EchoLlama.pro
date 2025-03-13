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
    ../LlamaClient.cpp

HEADERS += \
    EchoLlama.h \
    FontAwesome.h \
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

# Configuration for different platforms
win32: {
    DEFINES += WIN32
}
macx: {
    DEFINES += MACX
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


