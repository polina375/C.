#pragma once
#include <vector>
#include <cmath>
#include <concepts>
#include "types.h"

namespace Neural {

    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    template<Arithmetic T>
    class NeuralNetwork {
    public:
        // Конструктор: hiddenSize – число нейронов в скрытом слое
        explicit NeuralNetwork(int hiddenSize = 4)
            : hiddenSize(hiddenSize)
            , w_ih(2 * hiddenSize)  // веса вход->скрытый
            , b_h(hiddenSize)       // смещения скрытого
            , w_ho(hiddenSize)      // веса скрытый->выход
            , b_o(0)                // смещение выхода
        {
            // Инициализация случайными числами в [-0.5, 0.5]
            for (auto& w : w_ih) w = rand() / (T(RAND_MAX) + 1) - T(0.5);
            for (auto& b : b_h) b = rand() / (T(RAND_MAX) + 1) - T(0.5);
            for (auto& w : w_ho) w = rand() / (T(RAND_MAX) + 1) - T(0.5);
            b_o = rand() / (T(RAND_MAX) + 1) - T(0.5);
        }

        // Прямой проход (кэширует z_h, a_h, z_o, a_o)
        T forward(T x, T y) {
            // Скрытый слой
            z_h.resize(hiddenSize);
            a_h.resize(hiddenSize);
            for (int i = 0; i < hiddenSize; ++i) {
                z_h[i] = w_ih[2 * i] * x + w_ih[2 * i + 1] * y + b_h[i];
                a_h[i] = sigmoid(z_h[i]);
            }
            // Выходной слой
            z_o = 0;
            for (int i = 0; i < hiddenSize; ++i) z_o += w_ho[i] * a_h[i];
            z_o += b_o;
            a_o = sigmoid(z_o);
            return a_o;
        }

        T forward(const Point2D& p) { return forward(T(p.x), T(p.y)); }

        // Предсказание класса (0 или 1)
        int predictClass(const Point2D& p, T threshold = T(0.5)) const {
            // Временный расчёт (не меняет кэш, т.к. метод const)
            std::vector<T> a_h_tmp(hiddenSize);
            for (int i = 0; i < hiddenSize; ++i) {
                T z = w_ih[2 * i] * T(p.x) + w_ih[2 * i + 1] * T(p.y) + b_h[i];
                a_h_tmp[i] = sigmoid(z);
            }
            T z = 0;
            for (int i = 0; i < hiddenSize; ++i) z += w_ho[i] * a_h_tmp[i];
            z += b_o;
            return sigmoid(z) >= threshold ? 1 : 0;
        }

        // Доступ к кэшу (для обратного распространения)
        const std::vector<T>& getHiddenZ() const { return z_h; }
        const std::vector<T>& getHiddenA() const { return a_h; }
        T getOutputZ() const { return z_o; }
        T getOutputA() const { return a_o; }

        // Сохранение/загрузка весов (опционально)
        std::vector<T> getWeights() const {
            std::vector<T> all;
            all.insert(all.end(), w_ih.begin(), w_ih.end());
            all.insert(all.end(), b_h.begin(), b_h.end());
            all.insert(all.end(), w_ho.begin(), w_ho.end());
            all.push_back(b_o);
            return all;
        }

        void setWeights(const std::vector<T>& w) {
            size_t pos = 0;
            for (size_t i = 0; i < w_ih.size(); ++i) w_ih[i] = w[pos++];
            for (size_t i = 0; i < b_h.size(); ++i) b_h[i] = w[pos++];
            for (size_t i = 0; i < w_ho.size(); ++i) w_ho[i] = w[pos++];
            b_o = w[pos];
        }

    private:
        int hiddenSize;
        std::vector<T> w_ih;   // веса вход -> скрытый (2*hiddenSize)
        std::vector<T> b_h;    // смещения скрытого слоя
        std::vector<T> w_ho;   // веса скрытый -> выход
        T b_o;                 // смещение выхода

        // Кэш для прямого прохода (изменяются в forward)
        std::vector<T> z_h;    // взвешенная сумма на скрытом слое
        std::vector<T> a_h;    // выход скрытого слоя (после сигмоиды)
        T z_o;                 // взвешенная сумма на выходе
        T a_o;                 // выход сети (после сигмоиды)

        static T sigmoid(T x) { return T(1) / (T(1) + std::exp(-x)); }
    };

} // namespace Neural
