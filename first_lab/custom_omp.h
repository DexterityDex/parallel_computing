#ifndef custom_omp.h
#define custom_omp.h
#include <thread>
#include <omp.h>

static unsigned g_thread_num = std::thread::hardware_concurrency();

unsigned get_num_threads() {
    return g_thread_num;
}

void set_num_threads(unsigned T) {
    g_thread_num = T;
    omp_set_num_threads(T);
}
#endif