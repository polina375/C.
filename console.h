#pragma once
#include <iostream>

namespace Console {
    // Выводит информационное сообщение с префиксом [INFO].
    static void info(const char* msg) { std::cout << "[INFO] " << msg << "\n"; }
    // Выводит имя переменной и её значение типа float.
    static void value(const char* name, float val) { std::cout << name << ": " << val << "\n"; }
}; 
