#include "EchoLlama.h"

#include "LlamaClient.h"

#include <QVBoxLayout>
#include <QTextEdit>
#include <QScrollBar>
#include <QPlainTextEdit>
#include <QToolButton>
#include <QGuiApplication>
#include <QTimer>
#include <QDir>
#include <QCoreApplication>
#include <QDebug>

#include "FontAwesome.h"

EchoLlama::EchoLlama(QWidget *parent)
    : QWidget(parent), chatDisplay(new QTextEdit(this)), promptInput(new QPlainTextEdit(this)), sendButton(new QToolButton(this)) {

    QFont fa = FontAwesome::getFontAwesome();
    fa.setPointSize(20);
    sendButton->setFont(fa);
    sendButton->setText(QChar(0xf1d8));  // Paper plane icon
    sendButton->setToolTip("Send");
    sendButton->setCursor(Qt::PointingHandCursor);  // Make it look clickable

    // Set style for inputGroup
    inputGroup = new QWidget(this);
    inputGroup->setFixedHeight(75);

    // Initialize UI components
    chatDisplay->setReadOnly(true);
    chatDisplay->setMinimumHeight(160);
    //chatDisplay->setCurrentCharFormat()
    promptInput->setFixedHeight(46);
    promptInput->setMinimumHeight(46);


    // Apply the styles
    applyStyles();

    // Set placeholder text for promptInput
    promptInput->setPlaceholderText("Ask Anything");

    // Create a horizontal layout for the input group
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);
    inputLayout->addWidget(promptInput);
    // Create a horizontal layout for the send button
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(sendButton, 0, Qt::AlignRight);

    // Spacer to offset the button by -10 pixels to the left (after the button)
    QSpacerItem* spacer = new QSpacerItem(10, 0, QSizePolicy::Fixed, QSizePolicy::Minimum);
    buttonLayout->addItem(spacer); // This adds the spacer to the right of the button

    // Add the button layout to the main layout
    inputLayout->addLayout(buttonLayout);

    // Main layout
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(chatDisplay);
    layout->addWidget(inputGroup);

    // Connect signals to slots
    connect(promptInput, &QPlainTextEdit::textChanged, this, &EchoLlama::handleTextChange);
    connect(sendButton, &QToolButton::clicked, this, &EchoLlama::sendClicked);

    // Initialize Llama after a short delay
    QTimer::singleShot(200, this, &EchoLlama::initializeLlama);
}

EchoLlama::~EchoLlama() {
    delete llamaClient;
}

void EchoLlama::initializeLlama() {
#ifdef __APPLE__
    const char* enginePath = "llama.cpp/gguf-v0.4.0-3352-g855cd073/metal/libLlamaEngine.1.dylib";
#elif WIN32
    const char* enginePath = "D:/GameFusion/Applications/LlamaEngine/build-EchoLlama-Desktop_Qt_5_15_0_MSVC2019_64bit-Debug/debug/llama.cpp/gguf-v0.4.0-3477-ga800ae46/cuda/LlamaEngined.dll";
#else
    const char* enginePath = "LlamaEngine.so";
#endif
    qDebug() << "Binary path:" << QCoreApplication::applicationFilePath();
    //chatDisplay->append("Binary path:" + QCoreApplication::applicationFilePath()+"\n");

    llamaClient = LlamaClient::Create("CUDA", enginePath);
    if (!llamaClient) {
        chatDisplay->append(LlamaClient::GetCreateError().c_str());
        chatDisplay->append("Binary path:" + QCoreApplication::applicationFilePath()+"\n");
        return;
    }

    loadLlama();
}

bool EchoLlama::loadLlama() {
    QString homeDir = QDir::homePath();
#ifdef __APPLE__
    QString modelPath = homeDir + "/.cache/EchoLlama/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf";
#elif WIN32
    QString modelPath = "D:/llm-studio-models/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/qwen2.5-coder-3b-instruct-q4_k_m.gguf";
#endif
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
    //chatDisplay->append("Prompt: " + prompt + "\n");
    QTextCursor cursor = chatDisplay->textCursor();
    cursor.movePosition(QTextCursor::End);

    cursor.insertBlock(); // insert new block for the prompt

    // Create a QTextBlockFormat for the block formatting options
    QTextBlockFormat blockFormat;
    blockFormat.setLeftMargin(100); // Set the left margin to 20 pixels
    blockFormat.setTopMargin(10);
    blockFormat.setLineHeight(25, QTextBlockFormat::FixedHeight);
    //blockFormat.setLineSpacing(QTextBlockFormat::FixedHeight, 20.0); // Set line spacing to 20 pixels

    // Apply the block format to the current block
    cursor.setBlockFormat(blockFormat);

    QTextCharFormat format;
    format.setForeground(Qt::gray); // #c8a2c8 in RGB
    cursor.setBlockCharFormat(format);

    cursor.insertText(prompt);

    // Ensure the scroll bar is updated to the bottom
    chatDisplay->ensureCursorVisible();

    cursor.insertBlock(); // insert new block for the response

    cursor.movePosition(QTextCursor::End);
    chatDisplay->ensureCursorVisible();

    QGuiApplication::processEvents();

    generateResponse(prompt);
}

void EchoLlama::responseCallback(const char* msg, void* userData) {
    // Save current scrollbar position and check if it was at the bottom
    QScrollBar* scrollBar = chatDisplay->verticalScrollBar();
    bool wasAtBottom = scrollBar->value() == scrollBar->maximum();

    QTextCursor cursor = chatDisplay->textCursor();
    cursor.movePosition(QTextCursor::End);

    // Create a QTextBlockFormat for the block formatting options
    QTextBlockFormat blockFormat;
    blockFormat.setLeftMargin(0); // Set the left margin to 20 pixels
    blockFormat.setLineHeight(25, QTextBlockFormat::FixedHeight); // Set line spacing to 25 pixels
    // Apply the block format to the current block
    cursor.setBlockFormat(blockFormat);

    QTextCharFormat format;
    format.setForeground(Qt::white); // #c8a2c8 in RGB
    cursor.setBlockCharFormat(format);

    cursor.insertText(msg);

    // Set the modified cursor back to the text edit
    chatDisplay->setTextCursor(cursor);

    // Ensure the scroll bar is updated to the bottom
    cursor.movePosition(QTextCursor::End);
    chatDisplay->ensureCursorVisible();

    // Only auto-scroll if we were already at the bottom before adding text
    if (wasAtBottom) {
        chatDisplay->moveCursor(QTextCursor::End);
        chatDisplay->ensureCursorVisible();
    }

    // Force an update
    chatDisplay->update();
    QGuiApplication::processEvents();
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
            QGuiApplication::processEvents();
            promptInput->clear();
        }
    }
}

void EchoLlama::sendClicked() {
    // Handle the send button click
    processPrompt(promptInput->toPlainText());
    QGuiApplication::processEvents();
    promptInput->clear();
}

void applyModernScrollbarStyle(QTextEdit* textEdit) {
    // Get the vertical scrollbar specifically
    QScrollBar* verticalScrollBar = textEdit->verticalScrollBar();

    QString scrollbarStyle = R"(
        QScrollBar:vertical {
            border: none;
            background: transparent;
            width: 10px;
            margin: 0px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: darkgray;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
            background: none;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
    )";

    verticalScrollBar->setStyleSheet(scrollbarStyle);
}

void EchoLlama::applyStyles() {
    // Apply general styles
    this->setStyleSheet(
            "QWidget {"
            "outline: 0;"
            "background-color: #272931;"
            "}"
        );

    // Set style for chatDisplay
    chatDisplay->setStyleSheet(
            "QTextEdit {"
            "background-color: #272931;"
            "border-radius: 15px;"
            "   font-size: 16px;" // Increased font size
            "}"
        );

    inputGroup->setStyleSheet(
            "QWidget {"
            "background-color: #1c1e24;"
            "border-radius: 15px;"
            "}"
        );

    applyModernScrollbarStyle(chatDisplay);

    // Set style for promptInput
    promptInput->setStyleSheet(
            "QPlainTextEdit {"
            "   border: 0px solid darkgray;"
            "   background-color: #1c1e24;"
            "   border-radius: 0px;"
            "   font-size: 16px;" // Increased font size
            "}"
            "QPlainTextEdit::placeholder {"
            "   color: gray;"
            "}"
            "QPlainTextEdit[placeholderText] {"
                "   color: gray;"
                "}"
            "QScrollBar:vertical {"
            "   border: none;"
            "   background: transparent;"
            "   width: 10px;"
            "   margin: 0px 0px 0px 0px;"
            "}"
            "QScrollBar::handle:vertical {"
            "   background: darkgray;"
            "   border-radius: 5px;"
            "   min-height: 20px;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "   background: none;"
            "   height: 0px;"
            "}"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
            "   background: none;"
            "}"
        );



    sendButton->setStyleSheet(R"(
        QToolButton {
            color: white;
            background: transparent;
            border: none;
            }
        QToolButton:hover {
                color: #00AEEF;
            }
            QToolButton:pressed {
                color: #0077CC;
            }
        )");


    QTextCursor cursor = chatDisplay->textCursor();
    QTextCharFormat format;
    format.setForeground(Qt::gray); // #c8a2c8 in RGB
    chatDisplay->setCurrentCharFormat(format);

    cursor = promptInput->textCursor();
    promptInput->setCurrentCharFormat(format);


}

