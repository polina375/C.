#include "types.h"
#include "linear.h"
#include "dataset.h"
#include "evaluation.h"
#include "console.h"
#include "neuralnet.h"

int main() {

    Console::info("Laboratory Work: Linear Binary Classification");
     // Коэффициенты прямой для бинарной классификации: y = k*x + b
    float k = 0.75f;
    float b = -0.25f;

    Console::info("Generating dataset...");

    // Генерация 250 точек на плоскости с метками 0 или 1
    // Метка зависит от того, находится ли точка выше или ниже линии y = k*x + b
    auto data = Dataset::generate(250, k, b);


    // Матрица линейного преобразования 2x2
    // Применяется к координатам точек (для демонстрации работы с матрицами)
    float transform[2][2] = { {1.0f,0.1f},{0.05f,1.0f} };

    // Берём первую точку и применяем линейное преобразование
    float oldX = data[0].x;
    float oldY = data[0].y;

    auto newVec = Linear::transform(transform, oldX, oldY);


    // Матрица линейного преобразования 2x2
    // Применяется к координатам точек (для демонстрации работы с матрицами)
    Console::value("Original X", oldX);
    Console::value("Transformed X", newVec.x);

    // Вычисляем среднюю абсолютную ошибку
    float error = Evaluation::meanAbsError(data, k, b);
    Console::value("Mean Absolute Error", error);

    // Сохраняем CSV для визуализации
    if (Dataset::saveCSV(data, "../points.csv"))
        Console::info("Dataset saved to ../points.csv");


    //  Демонстрация нейронной сети (прямой проход) 
    Console::info(" Neural Network forward pass demo");

    // Создаём сеть: 4 нейрона в скрытом слое, тип float
    Neural::NeuralNetwork<float> net(4);

    // Берём первую точку из датасета и прогоняем через сеть
    Point2D testPoint = data[0];
    float probability = net.forward(testPoint);
    int predictedClass = net.predictClass(testPoint);

    Console::value("Test point x", testPoint.x);
    Console::value("Test point y", testPoint.y);
    Console::value("True label", testPoint.label);
    Console::value("Network probability", probability);
    Console::value("Predicted class", predictedClass);

    Console::info("Note: Network weights are random, no training yet.");

    return 0;
}
