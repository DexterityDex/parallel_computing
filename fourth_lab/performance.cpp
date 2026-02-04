#include "performance.h"
#include <memory>
#include <thread>
#include "num_threads.h"
#include "randomize.h"
#include "vector_mod.h"

std::vector<measurement> run_experiments()
{
    constexpr std::size_t word_count = (std::size_t(1) << 28) / sizeof(IntegerWord);
    constexpr IntegerWord divisor = INTWORD_MAX;
    auto data = std::make_unique<IntegerWord[]>(word_count);
    std::vector<measurement> results;
    randomize(data.get(), word_count * sizeof(IntegerWord));
    results.reserve(std::thread::hardware_concurrency());

    const int num_attempts = 10;

    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); ++T)
    {
        set_num_threads(T);

        IntegerWord final_result = 0;
        double total_time = 0.0;

        for (int attempt = 0; attempt < num_attempts; ++attempt)
        {
            using namespace std::chrono;
            auto tm0 = steady_clock::now();
            auto result = vector_mod(data.get(), word_count, divisor);
            auto time = duration_cast<milliseconds>(steady_clock::now() - tm0);

            total_time += time.count();

            if (attempt == num_attempts - 1)
            {
                final_result = result;
            }
        }

        results.emplace_back(measurement{final_result, std::chrono::milliseconds(static_cast<long>(total_time / num_attempts))});
    }

    return results;
}