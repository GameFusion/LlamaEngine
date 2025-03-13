#include "FontAwesome.h"
#include <QFontDatabase>
#include <QPainter>
#include <QPixmap>
#include <QDebug>

QFont FontAwesome::getFontAwesome() {
    static const QString fontPath(":/Resources/fonts/fa-solid-900.ttf"); // Corrected resource path
    int fontId = QFontDatabase::addApplicationFont(fontPath);
    
    if (fontId == -1) {
        qWarning() << "Failed to load FontAwesome from" << fontPath;
        return QFont(); // Return default font on failure
    }

    QStringList fontFamilies = QFontDatabase::applicationFontFamilies(fontId);
    if (fontFamilies.isEmpty()) {
        qWarning() << "Font loaded but no families found.";
        return QFont();
    }

    return QFont(fontFamilies.first());
}

QIcon FontAwesome::createIconFromFont(QChar character, const QSize &size, const QColor &color) {
    QPixmap pixmap(size);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    QFont font = getFontAwesome();
    font.setPointSize(size.width() / 2); // Adjust size dynamically

    painter.setFont(font);
    painter.setPen(color);
    painter.drawText(QRect(0, 0, size.width(), size.height()), Qt::AlignCenter, character);
    painter.end();

    return QIcon(pixmap);
}

void FontAwesome::setupToolButton(QToolButton *button, QChar iconChar) {
    QSize iconSize(32, 32);
    QIcon icon = createIconFromFont(iconChar, iconSize, Qt::white);
    button->setIcon(icon);
}

