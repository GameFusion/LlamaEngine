
#include "EchoLlama.h"

// Create a worker class for the thread
class ResponseWorker : public QObject {
    Q_OBJECT

public:
    ResponseWorker(EchoLlama* parent = nullptr) : QObject(nullptr), echoLlama(parent) {}

public slots:
    void processWithImage(const QString& prompt, const QString& imagePath) {
        echoLlama->generateResponse(prompt, imagePath);
        emit finished();
    }

    void processWithoutImage(const QString& prompt) {
        echoLlama->generateResponse(prompt);
        emit finished();
    }

signals:
    void finished();

private:
    EchoLlama* echoLlama;
};
