#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <iomanip>
#include <random>

#include "csv_functions.h"

#define cols 16384
#define rows 16384

void add_matrix(double* A, const double* B, const double* C, size_t total_elements) {
    for (size_t i = 0; i < total_elements; i++) {
        A[i] = B[i] + C[i];
    }
}

void add_matrix_avx2(double* C, const double* A, const double* B, size_t total_elements) {
    size_t i = 0;
    const size_t vec_size = 4;

    for (; i + vec_size <= total_elements; i += vec_size) {
        __m256d a = _mm256_loadu_pd(&A[i]);
        __m256d b = _mm256_loadu_pd(&B[i]);
        __m256d c = _mm256_add_pd(a, b);
        _mm256_storeu_pd(&C[i], c);
    }

    for (; i < total_elements; ++i) {
        C[i] = A[i] + B[i];
    }
}

/**
 * Скалярное умножение матриц в формате row-major.
 * A: rA x cA
 * B: rB x cB
 * C: rC x cC
 * Условие:  B (rA x cB), C (cB x cA), результат A (rA x cA)
 */
void mul_matrix(double* A, size_t rA, size_t cA,
                const double* B, size_t rB, size_t cB,
                const double* C, size_t rC, size_t cC) {
    // Проверяем корректность размеров
    assert(rA == rB);      // строки результата = строкам B
    assert(cA == cC);      // столбцы результата = столбцам C
    assert(cB == rC);      // внутренние размеры совпадают

    for (size_t i = 0; i < rA; ++i) {
        for (size_t j = 0; j < cA; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cB; ++k) {
                // B: [i][k], C: [k][j]
                sum += B[i * cB + k] * C[k * cC + j];
            }
            A[i * cA + j] = sum;
        }
    }
}

/**
 * AVX2-умножение матриц в формате row-major.
 * Векторизуем по 4 столбцам сразу.
 */
void mul_matrix_avx2(double* A,
                     size_t rA, size_t cA,
                     const double* B,
                     size_t rB, size_t cB,
                     const double* C,
                     size_t rC, size_t cC) {
    assert(rA == rB);
    assert(cA == cC);
    assert(cB == rC);

    const size_t vec_cols = 4;
    size_t j_vec_end = (cA / vec_cols) * vec_cols; // максимум кратное 4

    for (size_t i = 0; i < rA; ++i) {
        // Векторизованная часть по 4 столбцам
        for (size_t j = 0; j < j_vec_end; j += vec_cols) {
            __m256d sum = _mm256_setzero_pd();

            for (size_t k = 0; k < cB; ++k) {
                double b_ik = B[i * cB + k];
                __m256d b_vec = _mm256_set1_pd(b_ik);

                __m256d c_vec = _mm256_loadu_pd(&C[k * cC + j]);

                sum = _mm256_fmadd_pd(b_vec, c_vec, sum);
            }

            _mm256_storeu_pd(&A[i * cA + j], sum);
        }

        for (size_t j = j_vec_end; j < cA; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cB; ++k) {
                sum += B[i * cB + k] * C[k * cC + j];
            }
            A[i * cA + j] = sum;
        }
    }
}

std::vector<double> generate_permutation_matrix(std::size_t n) {
    std::vector<double> permut_matrix(n * n, 0.0);

    for (std::size_t i = 0; i < n; i++) {
        // единицы на побочной диагонали
        permut_matrix[i * n + (n - 1 - i)] = 1.0;
    }

    return permut_matrix;
}

void randomize_matrix(double* matrix, std::size_t matrix_order) {
    std::uniform_real_distribution<double> unif(0.0, 100000.0);
    std::mt19937 rng(std::random_device{}());

    for (std::size_t i = 0; i < matrix_order * matrix_order; i++) {
        matrix[i] = unif(rng);
    }
}

void print_matrix(const double* matrix, size_t colsc, size_t rowsc,
                  size_t max_rows = 5, size_t max_cols = 5) {
    std::cout << std::fixed << std::setprecision(2);
    for (size_t r = 0; r < std::min(rowsc, max_rows); ++r) {
        for (size_t c = 0; c < std::min(colsc, max_cols); ++c) {
            std::cout << matrix[r * colsc + c] << " ";
        }
        std::cout << (colsc > max_cols ? "... " : "") << "\n";
    }
    if (rowsc > max_rows)
        std::cout << "... \n";
    std::cout << "\n";
}

int mul_matrix_experiment() {
    const int num_attempts = 10;
    const std::size_t matrix_order = 16 * 4 * 9; // 576

    auto calculate_average_time =
        [](auto func,
           double* A,
           const double* B,
           const double* C,
           size_t matrix_order,
           int iterations)
    {
        double total_time = 0.0;

        for (int i = 0; i < iterations; i++)
        {
            // каждый раз новый A, чтобы не было прогрева на одинаковых данных
            randomize_matrix(A, matrix_order);

            auto t1 = std::chrono::steady_clock::now();
            func(A, matrix_order, matrix_order,
                 B, matrix_order, matrix_order,
                 C, matrix_order, matrix_order);
            auto t2 = std::chrono::steady_clock::now();

            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }
        return total_time / iterations;
    };

    std::vector<double> A(matrix_order * matrix_order),
                        B = generate_permutation_matrix(matrix_order),
                        C(matrix_order * matrix_order),
                        D(matrix_order * matrix_order);

    randomize_matrix(A.data(), matrix_order);

    std::cout << "Матрица A:\n";
    print_matrix(A.data(), matrix_order, matrix_order);

    std::cout << "Матрица B (перестановочная):\n";
    print_matrix(B.data(), matrix_order, matrix_order);

    // Проверка корректности
    mul_matrix(C.data(), matrix_order, matrix_order,
               A.data(), matrix_order, matrix_order,
               B.data(), matrix_order, matrix_order);

    mul_matrix_avx2(D.data(), matrix_order, matrix_order,
                    A.data(), matrix_order, matrix_order,
                    B.data(), matrix_order, matrix_order);

    if (std::memcmp(static_cast<const void*>(C.data()),
                    static_cast<const void*>(D.data()),
                    matrix_order * matrix_order * sizeof(double)) != 0) {
        std::cout << "Результат перемножения некорректен\n";
        return -1;
    }

    double avg_time_basic = calculate_average_time(
        mul_matrix, C.data(), A.data(), B.data(), matrix_order, num_attempts);
    std::cout << "Скалярное умножение: время = "
              << avg_time_basic << " мс, ускорение = 1\n";

    double avg_time_avx2 = calculate_average_time(
        mul_matrix_avx2, D.data(), A.data(), B.data(), matrix_order, num_attempts);
    std::cout << "Векторное умножение (AVX2): время = "
              << avg_time_avx2
              << " мс, ускорение = " << (avg_time_basic / avg_time_avx2) << "\n";

    return 0;
}

int sum_matrix() {
    size_t total_elements = cols * rows;
    std::vector<double> B(total_elements, 1.0),
                        C(total_elements, -1.0),
                        A(total_elements);

    auto calculate_average_time =
        [](auto func,
           double* A, const double* B, const double* C,
           size_t total_elements, int iterations)
    {
        double total_time = 0.0;
        for (int i = 0; i < iterations; i++)
        {
            auto t1 = std::chrono::steady_clock::now();
            func(A, B, C, total_elements);
            auto t2 = std::chrono::steady_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }
        return total_time / iterations;
    };

    const int num_attempts = 10;

    std::cout << "Матрица B:\n";
    print_matrix(B.data(), cols, rows);

    std::cout << "Матрица C:\n";
    print_matrix(C.data(), cols, rows);

    double avg_time_basic = calculate_average_time(
        add_matrix, A.data(), B.data(), C.data(), total_elements, num_attempts);
    std::cout << "Среднее время. Обычное сложение: "
              << avg_time_basic << " мс.\n";

    std::cout << "Результат обычного сложения:\n";
    print_matrix(A.data(), cols, rows);

    std::fill_n(A.data(), total_elements, 0.0);
    std::fill_n(B.data(), total_elements, 1.0);
    std::fill_n(C.data(), total_elements, -1.0);

    double avg_time_avx2 = calculate_average_time(
        add_matrix_avx2, A.data(), B.data(), C.data(), total_elements, num_attempts);
    std::cout << "Среднее время. AVX2: "
              << avg_time_avx2 << " мс.\n";

    std::cout << "Результат векторного сложения:\n";
    print_matrix(A.data(), cols, rows);

    return 0;
}

int main() {
    // return mul_matrix_experiment();
    return sum_matrix();
}
