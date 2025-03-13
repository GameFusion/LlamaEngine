#ifndef FontAwesome_H
#define FontAwesome_H

#include <QFont>
#include <QToolButton>
#include <QIcon>
#include <QSize>
#include <QColor>

class FontAwesome {
public:
    static QFont getFontAwesome();
    static QIcon createIconFromFont(QChar character, const QSize &size, const QColor &color = Qt::black);
    static void setupToolButton(QToolButton *button, QChar iconChar);
};

#endif // FontAwesome_H
