//
// Created by root on 8/1/25.
//

#include "../include/testSIMD.h"
#include <datatypes.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>
#include <immintrin.h> // for SIMD version
//#include <Eigen/Core>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h> // for OpenMP

using namespace Eigen;
using namespace std::chrono;

void crossProduct3x3_standard_loop(Tensor3f &A, Tensor3f &B, Tensor3f &C) {
    const auto& dims = A.dimensions();
    long rows = dims[0]; // height
    long cols = dims[1]; // width

    //TensorMap<Tensor<float, 2>> Aflat(A.data(), rows*cols, 3);
    //TensorMap<Tensor<float, 2>> Bflat(B.data(), rows*cols, 3);
    //TensorMap<Tensor<float, 2>> Cflat(C.data(), rows*cols, 3);
    //#pragma omp parallel for
    //for (long j = 0; j < rows*cols; ++j){
    //    Cflat(j, 0) = Aflat(j, 2) * Bflat(j, 1) - Aflat(j, 1) * Bflat(j, 2);  // y
    //    Cflat(j, 1) = Aflat(j, 0) * Bflat(j, 2) - Aflat(j, 2) * Bflat(j, 0);  // x
    //    Cflat(j, 2) = Aflat(j, 1) * Bflat(j, 0) - Aflat(j, 0) * Bflat(j, 1);  // z
    //}
    for (long i = 0; i < rows; ++i){
        for (long j = 0; j < cols; ++j){
            C(i, j, 0) = A(i, j, 2) * B(i, j, 1) - A(i, j, 1) * B(i, j, 2);  // y
            C(i, j, 1) = A(i, j, 0) * B(i, j, 2) - A(i, j, 2) * B(i, j, 0);  // x
            C(i, j, 2) = A(i, j, 1) * B(i, j, 0) - A(i, j, 0) * B(i, j, 1);  // z
        }
    }
}

void crossProduct3x3_partial_parallel(Tensor3f &A, Tensor3f &B, Tensor3f &C) {
    const auto& dims = A.dimensions();
    long rows = dims[0]; // height
    long cols = dims[1]; // width

    //TensorMap<Tensor<float, 2>> Aflat(A.data(), rows*cols, 3);
    //TensorMap<Tensor<float, 2>> Bflat(B.data(), rows*cols, 3);
    //TensorMap<Tensor<float, 2>> Cflat(C.data(), rows*cols, 3);
    //#pragma omp parallel for
    //for (long j = 0; j < rows*cols; ++j){
    //    Cflat(j, 0) = Aflat(j, 2) * Bflat(j, 1) - Aflat(j, 1) * Bflat(j, 2);  // y
    //    Cflat(j, 1) = Aflat(j, 0) * Bflat(j, 2) - Aflat(j, 2) * Bflat(j, 0);  // x
    //    Cflat(j, 2) = Aflat(j, 1) * Bflat(j, 0) - Aflat(j, 0) * Bflat(j, 1);  // z
    //}
    for (long i = 0; i < rows; ++i){
        #pragma omp parallel for
        for (long j = 0; j < cols; ++j){
            C(i, j, 0) = A(i, j, 2) * B(i, j, 1) - A(i, j, 1) * B(i, j, 2);  // y
            C(i, j, 1) = A(i, j, 0) * B(i, j, 2) - A(i, j, 2) * B(i, j, 0);  // x
            C(i, j, 2) = A(i, j, 1) * B(i, j, 0) - A(i, j, 0) * B(i, j, 1);  // z
        }
    }
}

void crossProduct3x3_parallel(Tensor2f &A, Tensor2f &B, Tensor2f &C) {
    const auto& dims = A.dimensions();
    long rows_cols = dims[0]; // height
    long depth = dims[1]; // width

    //TensorMap<Tensor<float, 2>> Aflat(A.data(), rows*cols, 3);
    //TensorMap<Tensor<float, 2>> Bflat(B.data(), rows*cols, 3);
    //TensorMap<Tensor<float, 2>> Cflat(C.data(), rows*cols, 3);
    //#pragma omp parallel for
    //for (long j = 0; j < rows*cols; ++j){
    //    Cflat(j, 0) = Aflat(j, 2) * Bflat(j, 1) - Aflat(j, 1) * Bflat(j, 2);  // y
    //    Cflat(j, 1) = Aflat(j, 0) * Bflat(j, 2) - Aflat(j, 2) * Bflat(j, 0);  // x
    //    Cflat(j, 2) = Aflat(j, 1) * Bflat(j, 0) - Aflat(j, 0) * Bflat(j, 1);  // z
    //}
    #pragma omp parallel for
    for (long i = 0; i < rows_cols; ++i){
        C(i, 0) = A(i, 2) * B(i,1) - A(i, 1) * B(i, 2);  // y
        C(i, 1) = A(i, 0) * B(i,2) - A(i, 2) * B(i, 0);  // x
        C(i, 2) = A(i, 1) * B(i,0) - A(i, 0) * B(i, 1);  // z
    }
}

void crossProduct3x3_SIMD_from_tensor(const Tensor3f& A,
                                      const Tensor3f& B,
                                      Tensor3f& C) {
    const auto& dims = A.dimensions();
    size_t rows = dims[0];
    size_t cols = dims[1];
    size_t count = rows * cols;

    // Allocate SoA arrays (aligned to 32). Using posix_memalign for portability.
    auto alloc_aligned = [](size_t n) {
        void* p = nullptr;
        posix_memalign(&p, 32, n * sizeof(float));
        return static_cast<float*>(p);
    };
    float* Ax = alloc_aligned(count);
    float* Ay = alloc_aligned(count);
    float* Az = alloc_aligned(count);
    float* Bx = alloc_aligned(count);
    float* By = alloc_aligned(count);
    float* Bz = alloc_aligned(count);
    float* Cx = alloc_aligned(count);
    float* Cy = alloc_aligned(count);
    float* Cz = alloc_aligned(count);

    // AoS -> SoA extraction
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            size_t idx = i * cols + j;
            Ax[idx] = A(i, j, 0);
            Ay[idx] = A(i, j, 1);
            Az[idx] = A(i, j, 2);
            Bx[idx] = B(i, j, 0);
            By[idx] = B(i, j, 1);
            Bz[idx] = B(i, j, 2);
        }

    // SIMD kernel (AVX2) - fall back to unaligned loads/stores (safe)
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 a0 = _mm256_load_ps(&Ax[i]); // assume aligned
        __m256 a1 = _mm256_load_ps(&Ay[i]);
        __m256 a2 = _mm256_load_ps(&Az[i]);

        __m256 b0 = _mm256_load_ps(&Bx[i]);
        __m256 b1 = _mm256_load_ps(&By[i]);
        __m256 b2 = _mm256_load_ps(&Bz[i]);

        __m256 cx = _mm256_sub_ps(_mm256_mul_ps(a1, b2), _mm256_mul_ps(a2, b1));
        __m256 cy = _mm256_sub_ps(_mm256_mul_ps(a2, b0), _mm256_mul_ps(a0, b2));
        __m256 cz = _mm256_sub_ps(_mm256_mul_ps(a0, b1), _mm256_mul_ps(a1, b0));

        _mm256_store_ps(&Cx[i], cx);
        _mm256_store_ps(&Cy[i], cy);
        _mm256_store_ps(&Cz[i], cz);
    }
    for (; i < count; ++i) {
        Cx[i] = Ay[i] * Bz[i] - Az[i] * By[i];
        Cy[i] = Az[i] * Bx[i] - Ax[i] * Bz[i];
        Cz[i] = Ax[i] * By[i] - Ay[i] * Bx[i];
    }

    // SoA -> AoS back into C
    for (size_t ii = 0; ii < rows; ++ii)
        for (size_t jj = 0; jj < cols; ++jj) {
            size_t idx = ii * cols + jj;
            C(ii, jj, 0) = Cx[idx];
            C(ii, jj, 1) = Cy[idx];
            C(ii, jj, 2) = Cz[idx];
        }

    free(Ax); free(Ay); free(Az);
    free(Bx); free(By); free(Bz);
    free(Cx); free(Cy); free(Cz);
}

void crossProduct3x3_tensor(const Tensor3f &A,
                            const Tensor3f &B,
                            Tensor3f &C) {
    // Expect shape (rows, cols, 3)
    assert(A.dimension(2) == 3);
    assert(B.dimension(0) == A.dimension(0) && B.dimension(1) == A.dimension(1) && B.dimension(2) == 3);
    assert(C.dimension(0) == A.dimension(0) && C.dimension(1) == A.dimension(1) && C.dimension(2) == 3);

    // Extract channels: chip(index, dimension) with dimension=2 (the last axis)
    auto Ax = A.chip(0, 2); // shape: rows x cols
    auto Ay = A.chip(1, 2);
    auto Az = A.chip(2, 2);

    auto Bx = B.chip(0, 2);
    auto By = B.chip(1, 2);
    auto Bz = B.chip(2, 2);

    // Compute cross-product components (broadcasting is identity here because shapes match)
    // Cx = Ay * Bz - Az * By
    Tensor2f Cx = (Ay * Bz) - (Az * By);
    // Cy = Az * Bx - Ax * Bz
    Tensor2f Cy = (Az * Bx) - (Ax * Bz);
    // Cz = Ax * By - Ay * Bx
    Tensor2f Cz = (Ax * By) - (Ay * Bx);

    // Assign back into C via chips
    C.chip(0, 2) = Cx;
    C.chip(1, 2) = Cy;
    C.chip(2, 2) = Cz;
}

// --- Utilities ---

float compute_checksum_tensor3(const Tensor3f& T) {
    auto dims = T.dimensions();
    long rows = dims[0], cols = dims[1];
    float acc = 0.0f;
    for (long i = 0; i < rows; ++i)
        for (long j = 0; j < cols; ++j)
            for (int k = 0; k < 3; ++k)
                acc += std::abs(T(i, j, k));
    return acc;
}

float compute_checksum_tensor2(const Tensor2f& T) {
    auto dims = T.dimensions();
    long n = dims[0];
    float acc = 0.0f;
    for (long i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            acc += std::abs(T(i, k));
    return acc;
}

// --- Benchmark functions (no lambdas) ---

void bench_tensor_variant(const Tensor3f& A, const Tensor3f& B, Tensor3f& C) {
    constexpr int repeats = 10;
    std::cout << "=== Benchmark: Eigen Tensor (crossProduct3x3_tensor) ===\n";
    double total_ms = 0.0;
    float checksum_acc = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        auto t0 = high_resolution_clock::now();
        crossProduct3x3_tensor(A, B, C);
        auto t1 = high_resolution_clock::now();
        duration<double, std::milli> dt = t1 - t0;
        total_ms += dt.count();
        float cs = compute_checksum_tensor3(C);
        checksum_acc += cs;
        std::cout << "  iter " << rep << ": " << dt.count() << " ms, checksum: " << cs << "\n";
    }
    std::cout << "  >>> average: " << (total_ms / repeats) << " ms, accumulated checksum: " << checksum_acc << "\n\n";
}

void bench_standard_loop_variant(Tensor3f& A, Tensor3f& B, Tensor3f& C) {
    constexpr int repeats = 10;
    std::cout << "=== Benchmark: standard nested loop ===\n";
    double total_ms = 0.0;
    float checksum_acc = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        auto t0 = high_resolution_clock::now();
        crossProduct3x3_standard_loop(A, B, C);
        auto t1 = high_resolution_clock::now();
        duration<double, std::milli> dt = t1 - t0;
        total_ms += dt.count();
        float cs = compute_checksum_tensor3(C);
        checksum_acc += cs;
        std::cout << "  iter " << rep << ": " << dt.count() << " ms, checksum: " << cs << "\n";
    }
    std::cout << "  >>> average: " << (total_ms / repeats) << " ms, accumulated checksum: " << checksum_acc << "\n\n";
}

void bench_partial_parallel_variant(Tensor3f& A, Tensor3f& B, Tensor3f& C) {
    constexpr int repeats = 10;
    std::cout << "=== Benchmark: partial parallel (inner j) ===\n";
    double total_ms = 0.0;
    float checksum_acc = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        auto t0 = high_resolution_clock::now();
        crossProduct3x3_partial_parallel(A, B, C);
        auto t1 = high_resolution_clock::now();
        duration<double, std::milli> dt = t1 - t0;
        total_ms += dt.count();
        float cs = compute_checksum_tensor3(C);
        checksum_acc += cs;
        std::cout << "  iter " << rep << ": " << dt.count() << " ms, checksum: " << cs << "\n";
    }
    std::cout << "  >>> average: " << (total_ms / repeats) << " ms, accumulated checksum: " << checksum_acc << "\n\n";
}

void bench_flat_parallel_variant(const Tensor2f& A_flat, const Tensor2f& B_flat, Tensor2f& C_flat) {
    constexpr int repeats = 10;
    std::cout << "=== Benchmark: flat parallel Tensor2f ===\n";
    double total_ms = 0.0;
    float checksum_acc = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        auto t0 = high_resolution_clock::now();
        crossProduct3x3_parallel(const_cast<Tensor2f&>(A_flat), const_cast<Tensor2f&>(B_flat), C_flat);
        auto t1 = high_resolution_clock::now();
        duration<double, std::milli> dt = t1 - t0;
        total_ms += dt.count();
        float cs = compute_checksum_tensor2(C_flat);
        checksum_acc += cs;
        std::cout << "  iter " << rep << ": " << dt.count() << " ms, checksum: " << cs << "\n";
    }
    std::cout << "  >>> average: " << (total_ms / repeats) << " ms, accumulated checksum: " << checksum_acc << "\n\n";
}

void bench_simd_variant(const Tensor3f& A, const Tensor3f& B, Tensor3f& C) {
    constexpr int repeats = 10;
    std::cout << "=== Benchmark: SIMD from tensor (manual SoA + AVX) ===\n";
    double total_ms = 0.0;
    float checksum_acc = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        auto t0 = high_resolution_clock::now();
        crossProduct3x3_SIMD_from_tensor(A, B, C);
        auto t1 = high_resolution_clock::now();
        duration<double, std::milli> dt = t1 - t0;
        total_ms += dt.count();
        float cs = compute_checksum_tensor3(C);
        checksum_acc += cs;
        std::cout << "  iter " << rep << ": " << dt.count() << " ms, checksum: " << cs << "\n";
    }
    std::cout << "  >>> average: " << (total_ms / repeats) << " ms, accumulated checksum: " << checksum_acc << "\n\n";
}

// --- Main ---

int main() {
    constexpr int rows = 1000;
    constexpr int cols = 1000;
    constexpr int channels = 3;
    const size_t count = static_cast<size_t>(rows) * cols;

    std::mt19937_64 rng(123456);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Allocate tensors A, B, C
    Tensor3f A(rows, cols, channels);
    Tensor3f B(rows, cols, channels);
    Tensor3f C(rows, cols, channels); // reusable output

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            for (int k = 0; k < channels; ++k) {
                A(i, j, k) = dist(rng);
                B(i, j, k) = dist(rng);
            }

    // Prepare flattened (count x 3) for flat parallel variant
    Tensor2f A_flat(count, 3);
    Tensor2f B_flat(count, 3);
    Tensor2f C_flat(count, 3);
    for (size_t idx = 0; idx < count; ++idx) {
        size_t i = idx / cols;
        size_t j = idx % cols;
        A_flat(idx, 0) = A(i, j, 0);
        A_flat(idx, 1) = A(i, j, 1);
        A_flat(idx, 2) = A(i, j, 2);
        B_flat(idx, 0) = B(i, j, 0);
        B_flat(idx, 1) = B(i, j, 1);
        B_flat(idx, 2) = B(i, j, 2);
    }

    // Warm-up each variant
    crossProduct3x3_tensor(A, B, C);
    crossProduct3x3_standard_loop(A, B, C);
    crossProduct3x3_partial_parallel(A, B, C);
    crossProduct3x3_parallel(A_flat, B_flat, C_flat);
    crossProduct3x3_SIMD_from_tensor(A, B, C);

    // Run benchmarks
    bench_tensor_variant(A, B, C);
    bench_standard_loop_variant(A, B, C);
    bench_partial_parallel_variant(A, B, C);
    bench_flat_parallel_variant(A_flat, B_flat, C_flat);
    bench_simd_variant(A, B, C);

    return 0;
}