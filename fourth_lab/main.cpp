#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>
#include "num_threads.h"
#include <fstream>

int main(int argc, char** argv)
{
    const int num_attempts = 10;
    std::ofstream output("long_arif.csv");

    output << "Threads,Value,Duration,Acceleration\n";

    for (std::size_t iTest = 1; iTest < test_data_count; ++iTest)
    {
        if (test_data[iTest].result != vector_mod(test_data[iTest].dividend, test_data[iTest].dividend_size, test_data[iTest].divisor))
        {
            std::cout << "ТЕСТЫ ПРОВАЛИЛИСЬ==\n";
            return -1;
        }
    }

    auto measurements = run_experiments();

    std::cout << "Результат выполнения:\n";
    std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |" << std::setw(3 + 2 * sizeof(IntegerWord)) << "Значение:" << " | "
              << std::setw(14) << "Длительность,мс:" << " | Ускорение:\n";

    for (std::size_t T = 1; T <= measurements.size(); ++T)
    {
        auto result = measurements[T - 1].result;
        auto duration = measurements[T - 1].time.count();
        auto acc = static_cast<double>(measurements[0].time.count()) / duration;

        // Вывод в консоль
        std::cout << std::setw(2) << T << " | 0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex << result;
        std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << duration;
        std::cout << " | " << acc << "\n";

        // Запись в файл
        output << T << ",";
        output << "0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex << result << ",";
        output << std::dec << duration << ",";
        output << acc << "\n";
    }

    output.close();
    return 0;
}
