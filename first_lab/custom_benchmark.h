#ifndef custom_becnchmark.h
#define custom_becnchmark.h
#include <omp.h>
#include <functional>


double get_omp_time(std::function<double(const double*, size_t)> fun,
                    const double* vector,
                    size_t N) {
    auto t1 = omp_get_wtime();
    fun(vector, N);
    return omp_get_wtime() - t1; 
}

template<std::invocable<const double*, size_t> F>
auto cpp_get_time(F fun, const double* V, size_t n) {
    using namespace std::chrono;
    auto t1 = steady_clock::now();
    fun(V, n);
    auto t2 = steady_clock::now();
    return duration_cast<milliseconds>(t2-t1).count();
}

template<typename V, typename R>
auto cpp_get_time_polymorph(std::function<R(const V*, size_t)> fun, const V* vector, size_t n) {
    using namespace std::chrono;
    auto t1 = steady_clock::now();
    fun(vector, n);
    auto t2 = steady_clock::now();
    return duration_cast<milliseconds>(t2-t1).count();
}

template<typename V, typename R, std::invocable<const V, size_t> F>
auto cpp_get_time_template(F fun, const V* vector, size_t n) {
    using namespace std::chrono;
    auto t1 = steady_clock::now();
    fun(vector, n);
    auto t2 = steady_clock::now();
    return duration_cast<milliseconds>(t2-t1).count();
}

class profiling_results_t {
    public:
    double result, time, speedup, effeciency;
    unsigned T;

    private:
    friend std::ostream& operator <<(std::ostream &os, const profiling_results_t& res) {
        os << res.result << "," << res.time << "," << res.speedup << "," << res.effeciency << "," << res.T;
        return os;
    }
};

template<class F>
auto run_experiment(F fun, const double* vector, size_t n) 
    requires std::is_invocable_r_v<double, F, const double*, std::size_t> {
    std::vector<profiling_results_t> r;
    std::size_t T_max = get_num_threads();
    for(std::size_t T = 1; T <= T_max; ++T) {
        set_num_threads(T);
        profiling_results_t rr;
        using namespace std::chrono;
        auto t1 = std::chrono::steady_clock::now();
        rr.result = fun(vector, n);
        auto t2 = std::chrono::steady_clock::now();
        rr.time = duration_cast<milliseconds>(t2 - t1).count();
        rr.T = T;
        r.push_back(rr); 
        r[T-1].speedup = r[0].time / r[T-1].time;
        r[T-1].effeciency = r[T-1].speedup / T;
    }
    return r;
}

#endif