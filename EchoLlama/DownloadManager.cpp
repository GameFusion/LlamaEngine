#include "DownloadManager.h"
#include "NetworkUtils.h"

#include <QDebug>


#include "DownloadManager.h"
#include "NetworkUtils.h"

DownloadManager::DownloadManager(QObject* parent)
    : QObject(parent)
{
}

void DownloadManager::downloadFile(const QString& url, QFile* file)
{
    if (activeDownloads.contains(url)) {
        qWarning() << "Download already in progress for" << url;
        return;
    }

    NetworkUtils* networkUtils = new NetworkUtils(this);
    activeDownloads[url] = networkUtils;

    connect(networkUtils, &NetworkUtils::progressUpdated, this, [this, url, networkUtils](qint64 bytesReceived, qint64 totalBytes) {
        emit progressUpdated(url, networkUtils->startOffset(), bytesReceived, totalBytes);
        });

    connect(networkUtils, &NetworkUtils::downloadFinished, this, [this, url]() {
        activeDownloads.remove(url);
        emit downloadFinished(url);
        });

    connect(networkUtils, &NetworkUtils::downloadError, this, [this, url](const QString& error) {
        activeDownloads.remove(url);
        emit downloadError(url, error);
        });

    networkUtils->downloadFile(url, file);
}

void DownloadManager::pauseDownload(const QString& url)
{
    if (activeDownloads.contains(url)) {
        NetworkUtils *networkUtils = activeDownloads[url];
        networkUtils->pauseDownload();
    }
}

bool DownloadManager::isActive(const QString& url)
{
    if (activeDownloads.contains(url)) {
        return true;
    }

    return false;
}

void DownloadManager::cancelDownload(const QString& url)
{
    if (activeDownloads.contains(url)) {
        NetworkUtils *networkUtils = activeDownloads[url];
        networkUtils->cancelDownload();
        delete networkUtils;

        emit downloadCancelled(url);
    }
}

