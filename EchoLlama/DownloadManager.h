#ifndef DownloadManager_H
#define DownloadManager_H

#include <QObject>
#include <QMap>
#include "NetworkUtils.h"

class QFile;

class DownloadManager : public QObject {
    Q_OBJECT

public:
    explicit DownloadManager(QObject* parent = nullptr);

    bool isActive(const QString& url);

    void downloadFile(const QString& url, QFile* file);
    void pauseDownload(const QString& url);
    void resumeDownload(const QString& url, QFile* file);
    void cancelDownload(const QString& url);

signals:
    void progressUpdated(const QString& url, qint64 startOffset, qint64 bytesReceived, qint64 totalBytes);
    void downloadFinished(const QString& url);
    void downloadError(const QString& url, const QString& error);
    void downloadCancelled(const QString& url);

private:
    struct DownloadTask {
        NetworkUtils* networkUtils;
        QFile* file;
    };

    QMap<QString, NetworkUtils*> activeDownloads;
};

#endif // DownloadManager_H
