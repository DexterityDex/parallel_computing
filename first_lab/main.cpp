#include "csv_functions.h"
#include "custom_omp.h"
#include "custom_benchmark.h"

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <vector>

double average_omp(const double* vector, size_t n) {
    double sum = 0.0;
    #pragma omp parallel
	{
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        for(size_t index = t; index < n; index+=T) {
            sum += vector[index];
        }
    }
    return sum / n;
}

double average_omp_with_array(const double* vector, size_t n) {
    double sum = 0.0;
    double* partical_sums = (double*) calloc(omp_get_num_procs(), sizeof(double));
    #pragma omp parallel
	{
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        for(size_t index = t; index < n; index+=T) {
            partical_sums[t] += vector[index];
        }
    }
    for(size_t i = 1; i < omp_get_num_procs(); ++i) {
        partical_sums[0] += partical_sums[i];
    }
    sum = partical_sums[0];
    delete partical_sums;
    return sum / n;
}

double average_omp_with_array_optimized(const double* vector, size_t n) {
    double sum = 0.0;
    double* partial_sums;
    unsigned T;
    #pragma omp parallel
	{
        unsigned t = omp_get_thread_num();
        #pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (double*) malloc(T * sizeof(vector[0]));
        }
        partial_sums[t] = 0.0;
        for(size_t index = t; index < n; index+=T) {
            partial_sums[t] += vector[index];
        }
    }
    for(size_t i = 1; i < omp_get_num_procs(); ++i) {
        partial_sums[0] += partial_sums[i];
    }
    sum = partial_sums[0];
    delete partial_sums;
    return sum / n;
}

double average_cpp_mtx(const double* vector, size_t n) {

    size_t T = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(T-1);
    std::mutex mtx;
    double res = 0.0;

    auto workers_proc = [&mtx, T, vector, n, &res] (unsigned t) {
        double partical = 0.0;
        for(size_t i = t; i < n; i += T) {
            partical += vector[i];
        }
        std::scoped_lock l {mtx};
        res += partical;
    };

    for (size_t t = 0; t < T; ++t) {
        workers.emplace_back(workers_proc, t);
    }

    for(auto &w: workers) {
        w.join();
    }
    return res / n;
}

double average_cpp_mtx_local(const double* vector, size_t n) {
    size_t T = std::thread::hardware_concurrency();
    std::vector<std::thread> workers(T-1);
    std::mutex mtx;
    double res = 0.0;
    auto workers_proc = [&mtx, T, vector, n, &res] (unsigned t) {
        double partical = 0.0;
        size_t b = n % T;
        size_t e = n / T;
        if(t < b) {
            b = t * ++e;
        } else {
            b += t * e;
        }
        e += b;
        for(size_t i = b; i < e; ++i) {
            partical += vector[i];
        }
        mtx.lock();
        res += partical;
        mtx.unlock();
    };

    for (size_t t = 0; t < T-1; ++t) {
        workers[t] = std::thread(workers_proc, t + 1);
    }

    for(auto &w: workers) {
        w.join();
    }
    return res / n;
}

struct partial_sum_t {
    alignas (64) double value;
};

double average_omp_with_struct(const double* vector, size_t n) {
    double sum = 0.0;
    unsigned T;
    partial_sum_t* partial_sums;
    #pragma omp parallel
	{
        unsigned t = omp_get_thread_num();
        #pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t*) malloc(T * sizeof(partial_sum_t));
        }
        partial_sums[t].value = 0;
        for(size_t index = t; index < n; index+=T) {
            partial_sums[t].value += vector[index];
        }
    }
    for(size_t i = 1; i < T; ++i) {
        partial_sums[0].value += partial_sums[i].value;
    }
    sum = partial_sums[0].value;
    delete partial_sums;
    return sum / n;
}

class barrier {
	unsigned current_barier_id = 0;
	const unsigned T_max;
	unsigned T;
	std::mutex mtx;
	std::condition_variable cv;
	public:

	barrier(unsigned threads): T_max(threads), T(T_max) {}

	void arrive_and_wait() {
		std::unique_lock lock {mtx};
		if(--T) {
			auto inner_id = current_barier_id;
			while(inner_id == current_barier_id) {
				cv.wait(lock);
			}
		}
		else {
			T = T_max;
			current_barier_id++;
			cv.notify_all();
		}
	}
};

double average_cpp_reduction(const double* vector, size_t N) {
	unsigned T = get_num_threads();
    double partical_results[T];
	std::vector<std::thread> workers;
	barrier bar(T);
	for(unsigned t = 0; t < T; ++t) {
		workers.emplace_back([&partical_results, &vector, &N, &bar, &T](unsigned t) {
			double partical = 0.0;
			size_t b = N % T;
			size_t e = N / T;
			if(t < b) {
				b = t * ++e;
			} else {
				b += t * e;
			}
			e += b;
			for(size_t i = b; i < e; ++i) {
				partical += vector[i];
			}
			partical_results[t] = partical;
			for(size_t step = 1, next = 2; step < T; step = next, next += next) {
				bar.arrive_and_wait();
				if((t &(next - 1)) == 0 && t + step < T) {
					partical_results[t] += partical_results[t + step];
				}
			}
		}, t);
	}
	for(unsigned t = 0; t < workers.size(); t++) {
		workers[t].join();
	}
	return partical_results[0] / N;
}

int main() {
    size_t N = 1u<<27;
	double* buf = new double[N];
	for(size_t i = 0; i < N; i++) {
		buf[i] = i;
	}
	std::string columns[5] = {"result", "time", "speedup", "effecency", "T"};
    auto result0 = run_experiment(average_omp_with_array, buf, N);
    auto result1 = run_experiment(average_omp_with_array_optimized, buf, N);
    auto result2 = run_experiment(average_cpp_mtx_local, buf, N);
    auto result3 = run_experiment(average_cpp_reduction, buf, N);

    write_to_csv(result0, "average_omp_with_array.csv", columns, 5);
    write_to_csv(result1, "average_omp_with_array_optimized.csv", columns, 5);
    write_to_csv(result2, "average_cpp_mtx_local.csv", columns, 5);
    write_to_csv(result3, "average_cpp_reduction.csv", columns, 5);

    delete buf;
}