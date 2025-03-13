#ifndef EchoLlama_h
#define EchoLlama_h

#include <QWidget>

class LlamaClient;
class QTextEdit;
class QPlainTextEdit;
class QToolButton;
class QComboBox;

/**
 * @file EchoLlama.h
 * @brief Defines the EchoLlama class for interacting with the LlamaEngine.
 * @details Manages model loading, text generation, and user interface interactions.
 * @author Andreas Carlen
 * @date March 6, 2025
 */

/**
 * @class EchoLlama
 * @brief Provides an interface to load and interact with a Llama model using a graphical user interface.
 */
class EchoLlama : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor for EchoLlama.
     * @param parent The parent widget.
     */
    explicit EchoLlama(QWidget *parent = nullptr);

    /**
     * @brief Destructor to clean up resources.
     */
    ~EchoLlama();

    /**
     * @brief Generates a response based on the given prompt.
     * @param prompt The input text prompt.
     */
    void generateResponse(const QString& prompt);

private slots:
    /**
     * @brief Initializes the Llama client and loads the model.
     */
    void initializeLlama();

    /**
     * @brief Handles changes in the text input to detect when a new prompt is submitted.
     */
    void handleTextChange();

    /**
     * @brief Processes the given prompt by appending it to the chat display and generating a response.
     * @param prompt The input text prompt.
     */
    void processPrompt(const QString& prompt);

    /**
     * @brief Slot to handle the send button click event.
     */
    void sendClicked();

    void handleArchitectureChange();
    void handleModelSelectionChange();
    void showModelInfo();
    void downloadModel();
    void showSettings();

private:
    /**
     * @brief Text edit widget for displaying chat messages.
     */
    QTextEdit* chatDisplay;

    /**
     * @brief Widget group containing input components (promptInput and sendButton).
     */
    QWidget* inputGroup;

    /**
     * @brief Plain text edit widget for user input prompts.
     */
    QPlainTextEdit* promptInput;

    /**
     * @brief Send button to process user input prompts.
     */
    QToolButton *sendButton;

    /**
     * @brief Pointer to the Llama client instance.
     */
    LlamaClient* llamaClient;

    QComboBox *modelSelectionComboBox;
    QComboBox *architectureComboBox;
    QToolButton *modelInfoButton;
    QToolButton *downloadButton;
    QToolButton *settingsButton;

    void loadCuratedModels();

    void setupUI();

    /**
     * @brief Callback function to handle incoming response messages.
     * @param msg The message content.
     * @param userData User data pointer (points to this EchoLlama instance).
     */
    void responseCallback(const char* msg, void* userData);

    /**
     * @brief Callback function to handle the completion of a response generation.
     * @param msg The message content.
     * @param userData User data pointer (points to this EchoLlama instance).
     */
    void finishedCallback(const char* msg, void* userData);

    /**
     * @brief Loads the Llama model with specified parameters.
     * @return True if the model loads successfully, false otherwise.
     */
    bool loadLlama();

    /**
     * @brief Applies styles to UI components (chatDisplay, inputGroup, promptInput, sendButton).
     */
    void applyStyles();
};

#endif // EchoLlama_h
