#include "NetworkUtils.h"
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QDebug>

//#include "Log.h"
//using GameFusion::Log;

NetworkUtils::NetworkUtils(QObject *parent)
    : QObject(parent),
    networkManager(new QNetworkAccessManager(this)),
    currentFile(nullptr),
    currentReply(nullptr),
    rangeStart(0)
{
}

void NetworkUtils::downloadFile(const QString &url, QFile *file)
{
    //Log().info() << "Downloading file from: "<<url.toUtf8().constData() <<"\n";

    QNetworkRequest request(url);
    qint64 fileSize = file->size();
    rangeStart = fileSize;

    if (fileSize > 0) {
        QString rangeValue = "bytes="+QString::number(fileSize) + "-";
        request.setRawHeader("Range", rangeValue.toUtf8() );
        //Log().info() << "Resuming download from byte "<<(int)fileSize<<"\n";
    }

    currentFile = file;
    currentReply = networkManager->get(request);

    connect(currentReply, &QNetworkReply::readyRead, this, &NetworkUtils::handleReadyRead);
    connect(currentReply, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::errorOccurred),
            this, &NetworkUtils::handleError);
    connect(currentReply, &QNetworkReply::downloadProgress,
            this, &NetworkUtils::handleDownloadProgress);
}

qint64 NetworkUtils::startOffset()
{
    return rangeStart;
}

void NetworkUtils::handleReadyRead()
{
    if (currentFile) {
        currentFile->write(currentReply->readAll());
    }
}

QString errorString(QNetworkReply::NetworkError code)
{
    switch (code) {
    case QNetworkReply::NoError:
        return "No error occurred.";
    case QNetworkReply::ConnectionRefusedError:
        return "The remote server refused the connection.";
    case QNetworkReply::RemoteHostClosedError:
        return "The remote host closed the connection prematurely, before any data was successfully received.";
    case QNetworkReply::HostNotFoundError:
        return "The remote host name was not found.";
    case QNetworkReply::TimeoutError:
        return "The connection to the remote server timed out.";
    case QNetworkReply::OperationCanceledError:
        return "The operation was canceled via calls to abort() or close().";
    case QNetworkReply::SslHandshakeFailedError:
        return "The SSL/TLS handshake failed and the encrypted channel could not be established.";
    case QNetworkReply::TemporaryNetworkFailureError:
        return "A temporary failure occurred, e.g., the network cable was unplugged temporarily.";
    case QNetworkReply::NetworkSessionFailedError:
        return "The connection was broken due to disconnection from the network. Please rejoin and try again.";
    case QNetworkReply::BackgroundRequestNotAllowedError:
        return "The background request is not allowed because application entered the suspended state.";
    case QNetworkReply::TooManyRedirectsError:
        return "Indicates that there were too many redirects.";
    case QNetworkReply::InsecureRedirectError:
        return "Indicates that there was a redirect to an insecure scheme (e.g., HTTP when HTTPS was used).";
    case QNetworkReply::UnknownNetworkError:
        return "An unknown network-related error was detected.";
    case QNetworkReply::ProxyConnectionRefusedError:
        return "The connection to the proxy server was refused.";
    case QNetworkReply::ProxyConnectionClosedError:
        return "The proxy server closed the connection prematurely.";
    case QNetworkReply::ProxyNotFoundError:
        return "The proxy host name was not found.";
    case QNetworkReply::ProxyTimeoutError:
        return "The connection to the proxy timed out or the proxy did not reply in time.";
    case QNetworkReply::ProxyAuthenticationRequiredError:
        return "The proxy requires authentication in order to establish a connection.";
    case QNetworkReply::ContentAccessDenied:
        return "The access to the remote content was denied (e.g., wrong credentials were supplied for authentication).";
    case QNetworkReply::ContentOperationNotPermittedError:
        return "A requested operation is not permitted on the given content, e.g., trying to write to a read-only file.";
    case QNetworkReply::ContentNotFoundError:
        return "The specified content was not found at the server (e.g., file or directory).";
    case QNetworkReply::AuthenticationRequiredError:
        return "The requested operation needs authentication but the credentials required were not provided.";
    case QNetworkReply::ContentReSendError:
        return "During data streaming, the remote host closed the connection prematurely, after which the client sent more data; this error can also be triggered by the server closing the connection before all the data was read.";
    case QNetworkReply::ProtocolUnknownError:
        return "The protocol specified in the URL is unknown.";
    case QNetworkReply::ProtocolInvalidOperationError:
        return "The requested operation is invalid for the given protocol.";
    default:
        return "An unknown network error occurred.";
    }
}

void NetworkUtils::handleError(QNetworkReply::NetworkError code)
{
    QString errorMessage = errorString(code);
    emit downloadError(errorMessage);
    qWarning() << "Download error:" << code;
    /*
    if (currentReply) {
        currentReply->deleteLater();
        currentReply = nullptr;
    }

    if (currentFile) {
        currentFile->close();
        currentFile = nullptr;
    }
    */
}

void NetworkUtils::handleDownloadProgress(qint64 bytesReceived, qint64 totalBytes)
{
    emit progressUpdated(bytesReceived, totalBytes);
    if ((bytesReceived) == totalBytes && currentReply) {
        currentReply->deleteLater();
        currentReply = nullptr;
        currentFile->close();
        currentFile = nullptr;
        emit downloadFinished();
    }

}

void NetworkUtils::cancelDownload()
{
    if (currentReply) {
        currentReply->abort();
        currentReply->deleteLater();
        currentReply = nullptr;
    }

    if (currentFile) {
        currentFile->close();
        currentFile->remove();
        currentFile->deleteLater();
        currentFile = nullptr;
    }
}

void NetworkUtils::pauseDownload()
{
    if (currentReply) {
        currentReply->abort();
        //bytesWritten = currentFile->pos(); // Save the position
        currentReply->deleteLater();
        currentReply = nullptr;
        currentFile->close();
    }
}
