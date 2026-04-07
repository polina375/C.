#pragma once
#include <iostream>

namespace Console {
    static void info(const char* msg) { std::cout << "[INFO] " << msg << "\n"; }
    static void value(const char* name, float val) { std::cout << name << ": " << val << "\n"; }
}; 
