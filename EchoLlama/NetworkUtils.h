#ifndef NetworkUtils_H
#define NetworkUtils_H

#include <QObject>
#include <QFile>
#include <QUrl>
#include <QNetworkReply>
#include <QNetworkAccessManager>

QT_BEGIN_NAMESPACE
class QNetworkAccessManager;
class QNetworkReply;
QT_END_NAMESPACE

class NetworkUtils : public QObject {
    Q_OBJECT

public:
    explicit NetworkUtils(QObject *parent = nullptr);
    void cancelDownload();
    void pauseDownload();

    qint64 startOffset();

signals:
    void progressUpdated(qint64 bytesReceived, qint64 totalBytes);
    void downloadFinished();
    void downloadError(const QString &errorMessage);

public slots:
    void downloadFile(const QString &url, QFile *file);

private slots:
    void handleReadyRead();
    void handleError(QNetworkReply::NetworkError code);
    void handleDownloadProgress(qint64 bytesReceived, qint64 totalBytes);
    
private:
    QNetworkAccessManager *networkManager;
    QFile *currentFile;
    QNetworkReply *currentReply;
    qint64 rangeStart;
};

#endif // NetworkUtils_H
