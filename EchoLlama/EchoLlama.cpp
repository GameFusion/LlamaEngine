#include "EchoLlama.h"

#include "LlamaClient.h"
#include "FontAwesome.h"
#include "DownloadManager.h"

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
#include <QProgressBar>
#include <QStandardPaths>
#include <QModelIndex>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QComboBox>
#include <QAbstractItemView>
#include <QHBoxLayout>

#include "llama_version.h"

bool systemPrompt=false;

EchoLlama::EchoLlama(QWidget *parent)
    : QWidget(parent), chatDisplay(new QTextEdit(this)), promptInput(new QPlainTextEdit(this)), sendButton(new QToolButton(this)){

    downloadManager = nullptr;
    llamaClient = nullptr;

    // Initialize UI components
    setupUI();



    // Apply the styles
    applyStyles();


    // Load curated models from JSON config file
    loadCuratedModels();
    //return;

    downloadManager = new DownloadManager(this);

    setupConnections();

    QGuiApplication::processEvents();

    // Initialize Llama after a short delay
    QTimer::singleShot(200, this, &EchoLlama::initializeLlama);
}

EchoLlama::~EchoLlama() {
    delete llamaClient;
}

void EchoLlama::loadCuratedModels() {
    qDebug() << "loadCuratedModels (1)!";

    QString modelsFilePath = ":/Resources/models.json";

    QFile file(modelsFilePath);
    if (!file.exists()) {
        qDebug() << "File not found!";
        return;
    }

    if (!file.open(QIODevice::ReadOnly)){
        qDebug() << "Failed to open file!";
        return;
    }

    qDebug() << "loadCuratedModels (2) models.json loaded!";

    QByteArray data = file.readAll();
    file.close();

    qDebug() << "loadCuratedModels (3) models.json read all!";


    // Create a QJsonParseError to capture any errors during parsing
    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (doc.isNull() || !doc.isArray()) {
        qDebug() << "Error parsing JSON at line" << parseError.errorString();
        return;
    }

    qDebug() << "loadCuratedModels (4) models.json to json doc";


    // Populate model selection dropdown
    modelsArray = doc.array();
    qDebug() << "loadCuratedModels (5) models.json to json array";

    for (int i = 0; i < modelsArray.size(); ++i) {
        QJsonObject modelObject = modelsArray[i].toObject();
        QString modelName = modelObject["name"].toString();

        if (!modelObject.contains("name") || !modelObject["name"].isString()) {
            qDebug() << "Error: Invalid model entry in JSON!";
            continue;
        }

        qDebug() << "loadCuratedModels (6) models.json found model " << modelName;

        modelSelectionComboBox->addItem(modelName);
    }

    qDebug() << "loadCuratedModels done";
    return;
}

void EchoLlama::setupUI() {

    QFont fa = FontAwesome::getFontAwesome();
    fa.setPointSize(20);

    //////////////////////////
    //
    // Setup the header top bar
    //

    // Architecture selection dropdown
    architectureComboBox = new QComboBox(this);

    // Model information icon button
    modelInfoButton = new QToolButton(this);
    modelInfoButton->setFont(fa);
    modelInfoButton->setText(QChar(0xf05a));  // Information icon
    modelInfoButton->setToolTip("Model Information");

    // Download icon button
    downloadButton = new QToolButton(this);
    downloadButton->setFont(fa);
    downloadButton->setText(QChar(0xf019));  // Download icon
    downloadButton->setToolTip("Download Model from Huging Faces");

    // Settings icon button
    settingsButton = new QToolButton(this);
    settingsButton->setFont(fa);
    settingsButton->setText(QChar(0xf013));  // Settings icon
    settingsButton->setToolTip("Model Settings");

    // Model selection dropdown
    modelSelectionComboBox = new QComboBox(this);




    // Top bar layout
    QHBoxLayout* topBarLayout = new QHBoxLayout();

    topBarLayout->setContentsMargins(0, 10, 0, 10);  // Remove all margins
    topBarLayout->addWidget(architectureComboBox);
    topBarLayout->addWidget(modelSelectionComboBox);
    topBarLayout->addWidget(downloadButton);
    topBarLayout->addWidget(settingsButton);
    topBarLayout->addWidget(modelInfoButton);

#ifdef __APPLE__
    architectureComboBox->addItem("Metal");
#else
    architectureComboBox->addItem("CUDA");
    architectureComboBox->addItem("Vulkan");
    architectureComboBox->addItem("CPU");
#endif

    // Set the size policy to allow shrinking
    architectureComboBox->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

    // Adjust the combo box width based on the content
    architectureComboBox->setMaximumWidth(architectureComboBox->view()->sizeHintForColumn(0) + 30);  // Add padding if needed

    progressBar = new QProgressBar(this);
    //////////////////////////
    //
    // Setup the chat area
    //

    sendButton->setFont(fa);
    sendButton->setText(QChar(0xf1d8));  // Paper plane icon
    sendButton->setToolTip("Send");
    sendButton->setCursor(Qt::PointingHandCursor);  // Make it look clickable


    // Create a horizontal layout for the send button
    QHBoxLayout* buttonLayout = new QHBoxLayout();

    // Setup attachment button
    setupAttachmentButton();

    // Spacer to offset the button by -10 pixels to the left (after the button)
    //QSpacerItem* spacer = new QSpacerItem(10, 0, QSizePolicy::Fixed, QSizePolicy::Minimum);
    //buttonLayout->addItem(spacer); // This adds the spacer to the right of the button

    buttonLayout->addStretch(1);

    // Add attachment button and send button to layout
    buttonLayout->addWidget(attachButton, 0, Qt::AlignRight);
    buttonLayout->addWidget(sendButton, 0, Qt::AlignRight);



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
    //QHBoxLayout* buttonLayout = new QHBoxLayout();
    //buttonLayout->addLayout(buttonLayout, 0, Qt::AlignRight);

    // Spacer to offset the button by -10 pixels to the left (after the button)
    //QSpacerItem* spacer = new QSpacerItem(10, 0, QSizePolicy::Fixed, QSizePolicy::Minimum);
    //buttonLayout->addItem(spacer); // This adds the spacer to the right of the button

    // Add the button layout to the main layout
    inputLayout->addLayout(buttonLayout);

    // Main layout
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(20, 0, 20, 20);
    layout->addLayout(topBarLayout);
    layout->addWidget(progressBar);
    layout->addWidget(chatDisplay);
    layout->addWidget(inputGroup);
}

void EchoLlama::setupConnections() {

    // Architecture selection dropdown connect
    connect(architectureComboBox, &QComboBox::currentIndexChanged, this, &EchoLlama::handleArchitectureChange);

    // Model information icon button
    connect(modelInfoButton, &QToolButton::clicked, this, &EchoLlama::showModelInfo);

    // Download icon button
    connect(downloadButton, &QToolButton::clicked, this, &EchoLlama::downloadModel);

    // Settings icon button
    connect(settingsButton, &QToolButton::clicked, this, &EchoLlama::showSettings);

    // Connect signals to slots
    connect(promptInput, &QPlainTextEdit::textChanged, this, &EchoLlama::handleTextChange);
    connect(sendButton, &QToolButton::clicked, this, &EchoLlama::sendClicked);

    connect(downloadManager, &DownloadManager::progressUpdated, this, &EchoLlama::updateDownloadProgress);
    connect(downloadManager, &DownloadManager::downloadFinished, this, &EchoLlama::onDownloadFinished);

    connect(modelSelectionComboBox, &QComboBox::currentIndexChanged, this, &EchoLlama::handleModelSelectionChange);
}

void EchoLlama::initializeLlama() {
    if (llamaClient)
        return;

    qDebug() << "initializeLlama";

    // Define base relative path
    const QString relativePath = "Resources/llama.cpp";

    // Determine resource base path based on OS
    QString resourceBasePath;
#ifdef __APPLE__
    resourceBasePath = QCoreApplication::applicationDirPath() + "/../" + relativePath;
#elif __linux__
    resourceBasePath = QCoreApplication::applicationDirPath() + "/../../" + relativePath;
#else
    resourceBasePath = QCoreApplication::applicationDirPath() + "/" + relativePath;
#endif

// Define library file names
#ifdef __APPLE__
    const QString libraryFileName = "libLlamaEngine.1.dylib";
#elif defined(WIN32)
    #ifdef DEBUG
       const QString libraryFileName = "LlamaEngined.dll";
    #else
        const QString libraryFileName = "LlamaEngine.dll";
    #endif
#else
    const QString libraryFileName = "LlamaEngine.so";
#endif

    // Construct full resource path
    const QString backendType = architectureComboBox->currentText().toLower();
    const QString localResourcePath = QString("%1/%2/%3/%4")
                                          .arg(resourceBasePath, LLAMA_COMMIT_VERSION, backendType, libraryFileName);

    qDebug() << "localResourcePath:" << localResourcePath;
    qDebug() << "Binary path:" << QCoreApplication::applicationFilePath();

    // Create the Llama client
    const QString arch = architectureComboBox->currentText();
    llamaClient = LlamaClient::Create(arch.toStdString(), localResourcePath.toStdString());

    if (!llamaClient) {
        chatDisplay->append(LlamaClient::GetCreateError().c_str());
        chatDisplay->append("Binary path: " + QCoreApplication::applicationFilePath() + "\n");
        return;
    }

    loadLlama();

    handleModelSelectionChange();

    QGuiApplication::processEvents();
}

QJsonObject EchoLlama::getSelectedModelObject() {
    int index = modelSelectionComboBox->currentIndex();

    if(index < 0){
        qDebug() << "No model found";
        return QJsonObject();
    }

    if (index >= modelsArray.count()) {
        qDebug() << "Model index out of bounds";
        return QJsonObject();
    }

    QJsonObject modelObject = modelsArray[index].toObject();
    if (!modelObject.contains("download_link")) {
        qDebug() << "Model missing download_link";
        return QJsonObject();
    }

    if (!modelObject.contains("byte_length")) {
        qDebug() << "Model missing byte_length";
        return QJsonObject();
    }

    return modelObject;
}


void displayImageInChat(QTextEdit* chatDisplay, const QString& imagePath) {
    // Load image from path
    QPixmap originalImage(imagePath);

    if (originalImage.isNull()) {
        // Handle image loading failure
        chatDisplay->append("Failed to load image: " + imagePath);
        return;
    }

    // Resize to thumbnail size (e.g., 100x100 pixels)
    QPixmap thumbnail = originalImage.scaled(
        QSize(100, 100),      // Target size
        Qt::KeepAspectRatio,  // Keep the aspect ratio
        Qt::SmoothTransformation  // Use smooth scaling
        );

    // Get document and cursor for insertion
    QTextDocument* document = chatDisplay->document();
    QTextCursor cursor(document);
    cursor.movePosition(QTextCursor::End);

    // Add the image to the document's resource collection
    document->addResource(
        QTextDocument::ImageResource,
        QUrl("file:" + imagePath),
        QVariant(thumbnail)
        );

    // Insert HTML that references the image
    cursor.insertHtml(QString("<img src='file:%1' />").arg(imagePath));

    // Add a line break after the image
    cursor.insertBlock();

    // Ensure the view scrolls to show the newly added content
    chatDisplay->ensureCursorVisible();
}

#include <QTextEdit>
#include <QString>
#include <QTextDocument>
#include <QTextCursor>
#include <QImage>
#include <QSize>
#include <QTextImageFormat>
#include <QTextBlockFormat>
#include <QTextImageFormat>
#include <random>

void displayMiniatureInChat(QTextEdit* chatDisplay, const QString& imagePath) {
    // Generate a random identifier
    std::random_device rd;
    std::mt19937 gen(rd());
    QString imageIdentifier = "miniature_" + QString::number(gen());

    // Load and scale down the image
    QImage image(imagePath);
    if (image.isNull()) {
        chatDisplay->append("Failed to load image: " + imagePath);
        return;
    }

    // Scale the image to a small size (adjust as needed)
    QImage scaledImage = image.scaled(
        QSize(256, 256),  // Small thumbnail size
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
        );

    // Get document for insertion
    QTextDocument* document = chatDisplay->document();

    // Add the scaled image to the document's resource collection
    document->addResource(
        QTextDocument::ImageResource,
        QUrl(imageIdentifier),
        QVariant(scaledImage)
        );

    // Get cursor at the end of the document
    QTextCursor cursor(document);
    cursor.movePosition(QTextCursor::End);

    // First, insert a new paragraph/block for the image
    QTextBlockFormat blockFormat;
    blockFormat.setAlignment(Qt::AlignLeft);  // Align to left margin
    blockFormat.setTopMargin(5);              // Add space above
    blockFormat.setBottomMargin(5);           // Add space below

    // Apply the block format to start a new paragraph
    cursor.insertBlock(blockFormat);

    // Create image format with proper spacing
    QTextImageFormat imageFormat;
    imageFormat.setName(imageIdentifier);
    imageFormat.setWidth(scaledImage.width());
    imageFormat.setHeight(scaledImage.height());

    // Add padding around the image
    //imageFormat.setMargin(5);  // 5 pixels of margin all around

    // Insert the image at cursor position (in the new paragraph)
    cursor.insertImage(imageFormat);

    // Insert another block after the image to ensure text starts in a new paragraph
    cursor.insertBlock(blockFormat);

    // Ensure the view scrolls to show the newly added content
    chatDisplay->ensureCursorVisible();
}

bool EchoLlama::loadLlama() {
    qDebug() << "loadLlama (1)";

    if(!llamaClient) {
        qDebug() << "loadLlama (2B) llama client is null";
        return false;
    }

    QJsonObject modelObject = getSelectedModelObject();
    if(modelObject.isEmpty())
        return false;

    QString downloadLink = modelObject["download_link"].toString();


    if(llamaClient && llamaClient->getModelFile() == downloadLink.toStdString())
        return true; // model already loaded

    QString homeDir = QDir::homePath();
    QString downloadFile = QUrl(downloadLink).fileName();
    QString modelPath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/.cache/EchoLlama/models";
    QString modelPathFile = QString("%1/%2").arg(modelPath).arg(downloadFile);
    QFile *file = new QFile(modelPathFile, this);

    qDebug() << "loadLlama (3) processing file" <<modelPathFile;

    if(file->exists()){

        qDebug() << "Model exists: " << modelPathFile;
        if(!modelObject.contains("byte_length")) {
            qDebug() << "loadLlama (5) missing bute_length in model attributes" <<modelPathFile;
            return false;
        }

        qint64 bytesTotal = modelObject["byte_length"].toInteger();
        qint64 fileSize = file->size();

        if(fileSize != bytesTotal){
            // download incomplete
            // here we display a message in the chat only
            // if a model is not already loaded
            if(!llamaClient->isModelLoaded())
                chatDisplay->append("Press the donwload to resume pulling this model for use\n");
            qDebug() << "Download incomplete: " << modelPathFile;
            return false;
        }
    }else{
        qDebug() << "Model not downloaded: " << modelPathFile;
        if(!llamaClient->isModelLoaded())
            chatDisplay->append("Press the donwload icon to use this model\n");
    }

    qDebug() << "loadLlama (7) setup model parameters: ";

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

    QFile modelFile(modelPathFile);
    if(!modelFile.exists()) {
        qDebug() << "EchoLlama model file does not exist: " << modelPathFile;
        return false;
    }

    bool success = llamaClient->loadModel(modelPathFile.toUtf8().constData(), params, paramCount);
    if (!success) {
        qDebug() << "loadLlama Failed to open model file: "<<modelPathFile;
        chatDisplay->append("Failed to open model file: \n"+modelPathFile+"\n");
        return false;
    }

    if(modelObject.contains("mmproj")){
        QString mmproj = modelObject["mmproj"].toString();

        if(!mmproj.isEmpty())
        {
            QString clipModelPathFile = QString("%1/%2").arg(modelPath).arg(mmproj);
            QTextEdit *chatDisplayPtr = (QTextEdit*)chatDisplay;  // Assuming chatDisplay is defined elsewhere
                llamaClient->loadClipModel(clipModelPathFile.toUtf8().constData(), [](const char* message, void *userData){
                    QTextEdit *display = (QTextEdit*)userData;
                    display->append("Loading clip model: "+QString(message));
                }, (void*)chatDisplay);
        }

        //displayMiniatureInChat(chatDisplay,  "/Users/andreascarlen/Documents/Screenshot 2024-03-22 at 12.09.51.png");
        //QGuiApplication::processEvents();
        //generateResponse("Hello, please describe this image!", "/Users/andreascarlen/Documents/Screenshot 2024-03-22 at 12.09.51.png");
        generateResponse("Hello");
    }
    else{

        systemPrompt = true;
        generateResponse("Hello!");
    }
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

    if (!attachedImagePath.isEmpty())
        // Process prompt with image
        generateResponse(prompt, attachedImagePath);
    else
        generateResponse(prompt);

    QGuiApplication::processEvents();
    promptInput->clear();
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

void EchoLlama::generateResponse(const QString& prompt, const QString &imagePath) {
    if (!llamaClient) {
        chatDisplay->append("Unable to generate response, Llama client not loaded.");
        return;
    }

    llamaClient->generateResponse(prompt.toUtf8().constData(), imagePath.toUtf8().constData(),
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
    if(systemPrompt){
        // system promps are generated directly without a user question directly by calling generateResponse vs processPromt
        // we need to add a new line
        QTextCursor cursor = chatDisplay->textCursor();
        cursor.movePosition(QTextCursor::End);
        cursor.insertBlock(); // insert new block for the prompt

        // pop system prompt status
        systemPrompt = false;
    }
}

void EchoLlama::handleTextChange() {
    QString text = promptInput->toPlainText();
    if (text.endsWith("\n")) {
        Qt::KeyboardModifiers modifiers = QGuiApplication::keyboardModifiers();
        if (!(modifiers & Qt::ShiftModifier)) {
            //text.chop(1);
            processPrompt(text);
        }
    }
}

void EchoLlama::sendClicked() {
    QString promptText = promptInput->toPlainText();

    // Normal text-only prompt
    processPrompt(promptText);
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
            "color: white;"
            "}"
        );

    progressBar->setStyleSheet(R"(
        QProgressBar {
            border: 0px solid #444;
            border-radius: 3px;
            background-color: #3d3f46;
            height: 6px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #0077CC;
            border-radius: 3px;
        }
    )");

    progressBar->setMaximumHeight(6);
    progressBar->setTextVisible(false); // Hide the percentage text

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

void EchoLlama::handleArchitectureChange(int index) {
    QString selectedArch = architectureComboBox->currentText();

    // prototype
    // Update the LlamaClient with the selected architecture
    //llamaClient.setArchitecture(selectedArch);
}

void EchoLlama::handleModelSelectionChange() {

    QString selectedModel = modelSelectionComboBox->currentText();

    QJsonObject modelObject = getSelectedModelObject();

    if (modelObject.isEmpty()) {
        return;
    }

    QString downloadLink = modelObject["download_link"].toString();
    if (downloadManager && downloadManager->isActive(downloadLink)){
        //chatDisplay->append("Model download in progress");
        return;
    }

    QString downloadFile = QUrl(downloadLink).fileName();
    QString modelPath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/.cache/EchoLlama/models";
    QString downloadFilePath = QString("%1/%2").arg(modelPath).arg(downloadFile);
    QFile *file = new QFile(downloadFilePath, this);

    qint64 bytesTotal = modelObject["byte_length"].toInteger();
    qint64 bytesDownloaded = 0;

    if(file->exists()){
        bytesDownloaded = file->size();

        if(bytesDownloaded < bytesTotal){
            //if(!llamaClient || !llamaClient->isModelLoaded())
            //    chatDisplay->append("Press the donwload icon to resume pull for this model");
            progressBar->show();
            updateProgress(bytesDownloaded, 0, bytesTotal);
        }else{
            progressBar->hide();
            updateProgress(0, bytesDownloaded, bytesDownloaded);
            if(llamaClient && !llamaClient->isModelLoaded())
                loadLlama();
            else{

                if(llamaClient && llamaClient->isModelLoaded() && llamaClient->getModelFile() != downloadFilePath.toStdString()){
                    delete llamaClient;
                    llamaClient = nullptr;
                    initializeLlama();
                }
            }
        }
    }
    else{
        //if(llamaClient && !llamaClient->isModelLoaded())
        //    chatDisplay->append("Press the donwload icon to get this model");
        updateProgress(0, 0, bytesTotal);
        progressBar->show();
    }

    // Prototype
    // Download the selected model
    //llamaClient.downloadModel(selectedModel);
}

void EchoLlama::showModelInfo() {
    QMessageBox::information(this, "Model Info", "Detailed information about the selected model.");
}

void EchoLlama::downloadModel() {
    QString selectedModel = modelSelectionComboBox->currentText();
    qDebug() << "Model selected: " << selectedModel;

    QJsonObject modelObject = getSelectedModelObject();
    if (modelObject.isEmpty()) {
        return;
    }

    QString downloadLink = modelObject["download_link"].toString();
    QString downloadFile = QUrl(downloadLink).fileName();
    QString modelPath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/.cache/EchoLlama/models";

    QDir dir(modelPath);
    if (!dir.exists())
        dir.mkpath(modelPath);
    QString downloadFilePath = QString("%1/%2").arg(modelPath).arg(downloadFile);
    QFile *file = new QFile(downloadFilePath, this);

    if (!file->open(QIODevice::Append)) {
        qWarning() << "Failed to open file for writing";
        return;
    }

    if(downloadManager)
        downloadManager->downloadFile(downloadLink, file);
    qDebug() << "Downloading Model "<< downloadLink << " to " << downloadFilePath;
}

void EchoLlama::showSettings() {
    QMessageBox::information(this, "Settings", "Configuration options for the LlamaEngine.");
}

void EchoLlama::updateProgress(qint64 starOffset, qint64 bytesReceived, qint64 totalBytes){
    // Calculate the percentage received
    int percentReceived = static_cast<int>((bytesReceived+starOffset) * 100 / (totalBytes+starOffset));
    progressBar->setValue(percentReceived);
}

void EchoLlama::updateDownloadProgress(const QString &url, qint64 starOffset, qint64 bytesReceived, qint64 totalBytes)
{
    QJsonObject modelObject = getSelectedModelObject();
    if (modelObject.isEmpty()) {
        return;
    }

    QString downloadLink = modelObject["download_link"].toString();

    if(downloadLink == url && totalBytes != 0)
    {
        updateProgress(starOffset, bytesReceived, totalBytes);

    }
}

void EchoLlama::onDownloadFinished(const QString &url)
{
    qDebug() << "Download complete";

    if (!llamaClient->isModelLoaded()) {
        //chatDisplay->append("Model download complete, loading llama...\n");
        QGuiApplication::processEvents();
        loadLlama();
    } else {
        QJsonObject modelObject = getSelectedModelObject();
        if (modelObject.isEmpty())
            return;

        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Model Downloaded",
                                      "Model download complete. Switch to the new model?",
                                      QMessageBox::Yes | QMessageBox::No);

        if (reply == QMessageBox::Yes) {
            // get json model for selecteed model
            for (const QJsonValue &modelValue : modelsArray) {

                QJsonObject modelObject = modelValue.toObject();
                QString downloadLink = modelObject["download_link"].toString();
                if(downloadLink == url){
                    QString modelName = modelObject.value("name").toString();
                    modelSelectionComboBox->setCurrentText(modelName);
                }

            }
        }
    }
}

void EchoLlama::setupAttachmentButton() {
    // Configure the attachment button using Font Awesome
    QFont fa = FontAwesome::getFontAwesome();
    fa.setPointSize(20);

    attachButton = new QToolButton(this);
    attachButton->setFont(fa);


    //attachButton->setText(QChar(0xf030));  // Font Awesome camera icon (or use 0xf0c6 for paperclip)
    attachButton->setText(QChar(0xf0c6));  // Font Awesome paperclip icon (or use 0xf0c6 for paperclip)

    attachButton->setToolTip("Attach an image");
    attachButton->setCursor(Qt::PointingHandCursor);

    // Connect the button to a slot that handles image selection
    connect(attachButton, &QToolButton::clicked, this, &EchoLlama::promptForImageFile);
}

#include <QFileDialog>

void EchoLlama::promptForImageFile() {
    // Open a file dialog to select an image
    QString imagePath = QFileDialog::getOpenFileName(
        this,
        "Select Image File",
        QDir::homePath(),
        "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)"
        );

    if (!imagePath.isEmpty()) {
        attachedImagePath = imagePath;



        // Let the user know an image is attached
        promptInput->setPlaceholderText("Promt the image...");

        // You could also add visual feedback that an image is attached
        attachButton->setStyleSheet("QToolButton { color: #00AEEF; }");

        displayMiniatureInChat(chatDisplay, imagePath);
        QGuiApplication::processEvents();
    }
}
