#include <QApplication>
#include "EchoLlama.h"

#ifdef WIN32
#include <qwindow>
#include <Windows.h>
#include <dwmapi.h> // For DwmSetWindowAttribute
#pragma comment(lib, "Dwmapi.lib")

void setWindowsTitleBarColor(HWND hwnd, COLORREF color) {
    // Set the title bar color
    DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, &color, sizeof(color));

    // Optionally set the text color on the title bar
    COLORREF textColor = RGB(255, 255, 255); // white text color
    DwmSetWindowAttribute(hwnd, DWMWA_TEXT_COLOR, &textColor, sizeof(textColor));
}
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    EchoLlama echo;
    echo.setWindowTitle("EchoLlama");
    echo.resize(600, 400);
    echo.show();

#ifdef WIN32
    QColor titleBarColor("#272931");
    HWND hwnd = (HWND)echo.windowHandle()->winId();

    // Set the title bar color (RGB color, e.g., dark gray)
    setWindowsTitleBarColor(hwnd, RGB(28, 30, 36)); // equivalent to #1c1e24
#endif

    return app.exec();
}

