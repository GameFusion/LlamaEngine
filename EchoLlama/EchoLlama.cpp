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
#include <QMessageBox>

#include "FontAwesome.h"

EchoLlama::EchoLlama(QWidget *parent)
    : QWidget(parent), chatDisplay(new QTextEdit(this)), promptInput(new QPlainTextEdit(this)), sendButton(new QToolButton(this)) {

    // Initialize UI components
    setupUI();

    // Apply the styles
    applyStyles();

    // Load curated models from JSON config file
    loadCuratedModels();

    // Connect signals to slots
    connect(promptInput, &QPlainTextEdit::textChanged, this, &EchoLlama::handleTextChange);
    connect(sendButton, &QToolButton::clicked, this, &EchoLlama::sendClicked);

    // Initialize Llama after a short delay
    QTimer::singleShot(200, this, &EchoLlama::initializeLlama);
}

EchoLlama::~EchoLlama() {
    delete llamaClient;
}

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QComboBox>

void EchoLlama::loadCuratedModels() {

    QString modelsFilePath = "/Users/andreascarlen/GameFusion/Applications/LlamaEngine/EchoLlama/Resources/models.json";
    QFile file(modelsFilePath);
    if (!file.exists()) {
        qDebug() << "File not found!";
        return;
    }

    if (!file.open(QIODevice::ReadOnly)){
        qDebug() << "Failed to open file!";
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    //QJsonDocument doc = QJsonDocument::fromJson(file.readAll());

    // Create a QJsonParseError to capture any errors during parsing
    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);

    if (doc.isNull() || !doc.isArray()) {
        qDebug() << "Error parsing JSON at line" << parseError.errorString();

        return;
    }

    //QJsonObject jsonObject = doc.array();
    QJsonArray modelsArray = doc.array();

    // Populate model selection dropdown
    for (int i = 0; i < modelsArray.size(); ++i) {
        QJsonObject modelObject = modelsArray[i].toObject();
        QString modelName = modelObject["name"].toString();
        modelSelectionComboBox->addItem(modelName);
    }
}

#include <QAbstractItemView>

void EchoLlama::setupUI() {

    QFont fa = FontAwesome::getFontAwesome();
    fa.setPointSize(20);

    //////////////////////////
    //
    // Setup the header top bar
    //

    // Architecture selection dropdown
    architectureComboBox = new QComboBox(this);
    connect(architectureComboBox, &QComboBox::currentIndexChanged, this, &EchoLlama::handleArchitectureChange);

    // Model information icon button
    modelInfoButton = new QToolButton(this);
    modelInfoButton->setFont(fa);
    modelInfoButton->setText(QChar(0xf05a));  // Information icon
    modelInfoButton->setToolTip("Model Information");
    connect(modelInfoButton, &QToolButton::clicked, this, &EchoLlama::showModelInfo);

    // Download icon button
    downloadButton = new QToolButton(this);
    downloadButton->setFont(fa);
    downloadButton->setText(QChar(0xf019));  // Download icon
    downloadButton->setToolTip("Download Model from Huging Faces");
    connect(downloadButton, &QToolButton::clicked, this, &EchoLlama::downloadModel);

    // Settings icon button
    settingsButton = new QToolButton(this);
    settingsButton->setFont(fa);
    settingsButton->setText(QChar(0xf013));  // Settings icon
    settingsButton->setToolTip("Model Settings");
    connect(settingsButton, &QToolButton::clicked, this, &EchoLlama::showSettings);

    // Model selection dropdown
    modelSelectionComboBox = new QComboBox(this);
    connect(modelSelectionComboBox, &QComboBox::currentIndexChanged, this, &EchoLlama::handleModelSelectionChange);

    // Top bar layout
    QHBoxLayout* topBarLayout = new QHBoxLayout();

    topBarLayout->setContentsMargins(0, 10, 0, 10);  // Remove all margins
    topBarLayout->addWidget(architectureComboBox);
    topBarLayout->addWidget(modelSelectionComboBox);
    topBarLayout->addWidget(downloadButton);
    topBarLayout->addWidget(settingsButton);
    topBarLayout->addWidget(modelInfoButton);

    architectureComboBox->addItem("CPU");
    architectureComboBox->addItem("CUDA");
    architectureComboBox->addItem("Vulkan");
    architectureComboBox->addItem("Metal");

    // Set the size policy to allow shrinking
    architectureComboBox->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

    // Adjust the combo box width based on the content
    architectureComboBox->setMaximumWidth(architectureComboBox->view()->sizeHintForColumn(0) + 30);  // Add padding if needed

    //////////////////////////
    //
    // Setup the chat area
    //

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
    layout->setContentsMargins(20, 0, 20, 20);
    layout->addLayout(topBarLayout);
    layout->addWidget(chatDisplay);
    layout->addWidget(inputGroup);


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

void applyStylesToWidgets(QLayout* layout) {
    if (!layout) return;

    // Loop through all items in the layout
    for (int i = 0; i < layout->count(); ++i) {
        // Get the layout item
        QLayoutItem* item = layout->itemAt(i);
        if (!item) continue;

        // Check if the item is a layout
        if (QLayout* subLayout = item->layout()) {
            // Recursive call for nested layouts
            applyStylesToWidgets(subLayout);
        } else {
            // It's a widget (QWidget)
            QWidget* widget = item->widget();
            if (widget) {
                // Apply style to QToolButton
                if (QToolButton* toolButton = qobject_cast<QToolButton*>(widget)) {
                    toolButton->setStyleSheet(R"(
                        QToolButton {
                            color: #929292;
                            background: transparent;
                            border: none;
                            font-size: 16px;
                        }
                        QToolButton:hover {
                            color: #00AEEF;
                        }
                        QToolButton:pressed {
                            color: #0077CC;
                        }
                    )");
                }
                // Apply style to QComboBox
                else if (QComboBox* comboBox = qobject_cast<QComboBox*>(widget)) {
                    comboBox->setStyleSheet(R"(
                        QComboBox {
                            border: none;
                            background: transparent;
                            padding-left: 6px;
                            color: #929292;
                            font-size: 12px;
                        }
                    QComboBox::drop-down {
                            subcontrol-position:  left;
                        }
                    )");
                }
            }
        }
    }
}

void EchoLlama::applyStyles() {


    applyStylesToWidgets(this->layout());

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
    format.setForeground(QColor(230, 230, 230)); // #c8a2c8 in RGB
    promptInput->setCurrentCharFormat(format);


}

void EchoLlama::handleArchitectureChange() {
    QString selectedArch = architectureComboBox->currentText();
    // Update the LlamaClient with the selected architecture
    //llamaClient.setArchitecture(selectedArch);
}

void EchoLlama::handleModelSelectionChange() {
    QString selectedModel = modelSelectionComboBox->currentText();
    // Load and configure the selected model
    //llamaClient.loadModel(selectedModel);

    // TODO
    // check if model exists and has been downloaded
    // model file should be in <home>./cache/EchoLlamma/<selectedModel>
    // if model file exists hide the download button
    // if model does not exist show download button

}

void EchoLlama::showModelInfo() {
    QMessageBox::information(this, "Model Info", "Detailed information about the selected model.");
}

void EchoLlama::downloadModel() {
    QString selectedModel = modelSelectionComboBox->currentText();
    // Download the selected model
    //llamaClient.downloadModel(selectedModel);
    qDebug() << "Model selected: " << selectedModel;
}

void EchoLlama::showSettings() {
    QMessageBox::information(this, "Settings", "Configuration options for the LlamaEngine.");
}


