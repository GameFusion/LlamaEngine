#include <QApplication>
#include "EchoLlama.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    EchoLlama echo;
    echo.setWindowTitle("EchoLlama");
    echo.resize(600, 400);
    echo.show();

    return app.exec();
}

