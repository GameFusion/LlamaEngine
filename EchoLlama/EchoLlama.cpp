#include "EchoLlama.h"

#include "LlamaClient.h"

#include <QVBoxLayout>
#include <QTextEdit>
#include <QScrollBar>
#include <QPlainTextEdit>
#include <QGuiApplication>
#include <QTimer>
#include <QDir>
#include <QCoreApplication>
#include <QDebug>

EchoLlama::EchoLlama(QWidget *parent)
    : QWidget(parent), chatDisplay(new QTextEdit(this)), promptInput(new QPlainTextEdit(this)) {

    chatDisplay->setReadOnly(true);
    chatDisplay->setMinimumHeight(150);

    promptInput->setFixedHeight(48);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(chatDisplay);
    layout->addWidget(promptInput);

    connect(promptInput, &QPlainTextEdit::textChanged, this, &EchoLlama::handleTextChange);

    QTimer::singleShot(200, this, &EchoLlama::initializeLlama);
}

EchoLlama::~EchoLlama() {
    delete llamaClient;
}

void EchoLlama::initializeLlama() {
#ifdef __APPLE__
    const char* enginePath = "llama.cpp/gguf-v0.4.0-3352-g855cd073/metal/libLlamaEngine.1.dylib";
#elif WIN32
    const char* enginePath = "LlamaEngine.dll";
#else
    const char* enginePath = "LlamaEngine.so";
#endif
    qDebug() << "Binary path:" << QCoreApplication::applicationFilePath();
    chatDisplay->append("Binary path:" + QCoreApplication::applicationFilePath()+"\n");

    llamaClient = LlamaClient::Create("CUDA", enginePath);
    if (!llamaClient) {
        chatDisplay->append(LlamaClient::GetCreateError().c_str());
        return;
    }

    loadLlama();
}

bool EchoLlama::loadLlama() {
    QString homeDir = QDir::homePath();
    QString modelPath = homeDir + "/.cache/EchoLlama/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf";
    int contextSize = 4096;
    float temperature = 0.7f;
    float topK = 40;
    float topP = 0.6;
    float repetitionPenalty = 1.2;

    ModelParameter params[] = {
        {"temperature", PARAM_FLOAT, &temperature},
        {"context_size", PARAM_INT, &contextSize},
        {"top_k", PARAM_FLOAT, &topK},
        {"top_P", PARAM_FLOAT, &topP},
        {"repetition_penalty", PARAM_FLOAT, &repetitionPenalty}
    };

    size_t paramCount = sizeof(params) / sizeof(params[0]);
    bool success = llamaClient->loadModel(modelPath.toUtf8().constData(), params, paramCount);
    if (!success) {
        chatDisplay->append("Failed to open model file: \n"+modelPath+"\n");
        return false;
    }

    qDebug() << "Model loaded successfully.";

    generateResponse("Hello!");
    promptInput->setFocus();

    return true;
}

void EchoLlama::processPrompt(const QString& prompt) {
    chatDisplay->append("Prompt: " + prompt + "\n");
    generateResponse(prompt);
}

void EchoLlama::generateResponse(const QString& prompt) {
    if (!llamaClient) {
        chatDisplay->append("Unable to generate response, Llama client not loaded.");
        return;
    }

    llamaClient->generateResponse(prompt.toUtf8().constData(),
          [](const char* msg, void* userData) {
              ((EchoLlama*)userData)->responseCallback(msg, userData);
          },
          [](const char* msg, void* userData) {
              ((EchoLlama*)userData)->finishedCallback(msg, userData);
          },
        this
    );
}

void EchoLlama::responseCallback(const char* msg, void* userData) {
    QTextCursor cursor = chatDisplay->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(msg);
    QGuiApplication::processEvents();
}

void EchoLlama::finishedCallback(const char* msg, void* userData) {
    // Optional: Handle any cleanup after response is finished
}

void EchoLlama::handleTextChange() {
    QString text = promptInput->toPlainText();
    if (text.endsWith("\n")) {
        Qt::KeyboardModifiers modifiers = QGuiApplication::keyboardModifiers();
        if (!(modifiers & Qt::ShiftModifier)) {
            text.chop(1);
            processPrompt(text);
            promptInput->clear();
        }
    }
}

