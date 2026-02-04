#include <barrier>
#include <bit>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

static unsigned nibble[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

unsigned flip_b(unsigned byte) {
    return (nibble[byte & 15] << 4) | nibble[(byte >> 4) & 15];
}

unsigned flip_s(unsigned v) {
    return flip_b(v & 0xFF) << 8 | flip_b(v >> 8);
}

unsigned flip_i(unsigned v) {
    return flip_s(v & 0xFFFF) << 16 | flip_s(v >> 16);
}

unsigned long long flip_ll(unsigned long long v) {
    return (unsigned long long) flip_i(v & 0xFFFFFFFF) << 32 | flip_i(v >> 32);
}

void bit_shuffle(const std::complex<double>* inp, std::complex<double>* out, std::size_t n)
{
    std::size_t shift = std::countl_zero<std::size_t>(n) + 1llu;
    for (std::size_t i = 0; i < n; i++)
    {
        out[flip_ll(i) >> shift] = inp[i];
    }
}

struct thread_range
{
    std::size_t b, e;
};

thread_range thread_task_range(std::size_t task_count, std::size_t thread_count, std::size_t thread_id)
{
    auto b = task_count % thread_count;
    auto s = task_count / thread_count;
    if (thread_id < b) b = ++s * thread_id;
    else b += s * thread_id;
    size_t e = b + s;
    return {b, e};
}

void fft_nonrec_multithreaded_core(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, int inverse, std::size_t thread_count)
{
    bit_shuffle(inp, out, n);

    std::barrier<> bar(thread_count);
    auto thread_lambda = [&out, n, inverse, thread_count, &bar](std::size_t thread_id){
        for (std::size_t group_length = 2; group_length <= n; group_length <<= 1)
        {
            if (thread_id + 1 <= n / group_length)
            {
                auto [b, e] = thread_task_range(n / group_length, thread_count, thread_id);
                for (std::size_t group = b; group < e; group++)
                {
                    for (std::size_t i = 0; i < group_length / 2; i++)
                    {
                        auto w = std::polar(1.0, -2 * std::numbers::pi_v<double> * i * inverse / group_length);
                        auto r1 = out[group_length * group + i];
                        auto r2 = out[group_length * group + i + group_length / 2];
                        out[group_length * group + i] = r1 + w * r2;
                        out[group_length * group + i + group_length / 2] = r1 - w * r2;
                    }
                }
            }

            bar.arrive_and_wait();
        }
    };

    std::vector<std::thread> threads(thread_count - 1);
    for (std::size_t i = 1; i < thread_count; i++)
    {
        threads[i - 1] = std::thread(thread_lambda, i);
    }
    thread_lambda(0);
    for (auto& i : threads)
    {
        i.join();
    }
}

void fft_nonrec_multithreaded(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, std::size_t thread_count)
{
    fft_nonrec_multithreaded_core(inp, out, n, 1, thread_count);
}

void ifft_nonrec_multithreaded(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, std::size_t thread_count)
{
    fft_nonrec_multithreaded_core(inp, out, n, -1, thread_count);
    for (std::size_t i = 0; i < n; i++)
    {
        out[i] /= static_cast<std::complex<double>>(n);
    }
}


int main()
{
    const std::size_t exp_count = 10;
    constexpr std::size_t n = 1llu << 20;
    std::vector<std::complex<double>> original(n), spectre(n), restored(n);

    auto print_vector = [](const std::vector<std::complex<double>>& v){
        for (std::size_t i = 0; i < v.size(); i++)
        {
            std::cout << "[" << i << "] " << std::fixed << v[i] << "\n";
        }
    };

    auto randomize_vector = [](std::vector<std::complex<double>>& v) {
        std::uniform_real_distribution<double> unif(0, 100000);
        std::default_random_engine re;
        for (std::size_t i = 0; i < v.size(); i++)
        {
            v[i] = unif(re);
        }
    };

    auto approx_equal = [](const std::vector<std::complex<double>>& v,
                           const std::vector<std::complex<double>>& u) {
        for (std::size_t i = 0; i < v.size(); i++)
        {
            if (std::abs(v[i] - u[i]) > 0.0001)
            {
                return false;
            }
        }
        return true;
    };

    std::ofstream output("furie.csv");

    output << "Threads,Duration,Acceleration\n";

    for (std::size_t i = 0; i < n / 2; i++)
    {
        original[i] = i;
        original[n - 1 - i] = i;
    }

    fft_nonrec_multithreaded(original.data(), spectre.data(), n, 4);
    ifft_nonrec_multithreaded(spectre.data(), restored.data(), n, 4);
    if (!approx_equal(original, restored))
    {
        std::cout << "ОШИБКА\n";
        return -1;
    }

    double time_sum_1;
    for (std::size_t i = 1; i <= std::thread::hardware_concurrency(); i++)
    {
        double time_sum = 0;

        for (std::size_t j = 0; j < exp_count; j++)
        {
            randomize_vector(original);
            auto t1 = std::chrono::steady_clock::now();
            fft_nonrec_multithreaded(original.data(), spectre.data(), n, i);
            auto t2 = std::chrono::steady_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }

        if (i == 1)
        {
            time_sum_1 = time_sum;
        }

	std::cout << std::setw(12) << i << " | "
                  << std::setw(12) << time_sum / exp_count << " | "
                  << std::setw(12) << (time_sum_1 / exp_count) / (time_sum / exp_count) <<"\n";

        output << i << "," << time_sum / exp_count << "," << (time_sum_1 / exp_count) / (time_sum / exp_count) << "\n";
    }

    output.close();
    return 0;
}