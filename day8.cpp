#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

/// Row-major 2D matrix of FP32 values stored in a flat 1D vector.
struct Matrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;

    /// Constructs a zero-initialized r x c matrix.
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    /// Returns a mutable reference to element (i, j) in row-major layout.
    float& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }

    /// Returns a read-only reference to element (i, j) in row-major layout.
    const float& operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
};

/// Row-major 2D matrix quantized to signed INT8 with a single per-tensor float scale.
struct QuantizedMatrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<std::int8_t> data;
    float scale;

    /// Constructs a zero-initialized r x c quantized matrix with the given scale.
    QuantizedMatrix(std::size_t r, std::size_t c, float s = 0.0f)
        : rows(r), cols(c), data(r * c, 0), scale(s) {}
};

/// Quantizes an FP32 matrix symmetrically into [-127, 127] using scale = max(|x|) / 127.
QuantizedMatrix quantize_symmetric(const Matrix& m) {
    float max_abs = 0.0f;
    for (float x : m.data) {
        const float a = std::fabs(x);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    if (max_abs == 0.0f) {
        return QuantizedMatrix(m.rows, m.cols, 0.0f);
    }

    const float scale = max_abs / 127.0f;
    const float inv_scale = 1.0f / scale;
    QuantizedMatrix qm(m.rows, m.cols, scale);
    for (std::size_t idx = 0; idx < m.data.size(); ++idx) {
        const float rounded = std::round(m.data[idx] * inv_scale);
        qm.data[idx] = static_cast<std::int8_t>(std::clamp(rounded, -127.0f, 127.0f));
    }
    return qm;
}

/// Raw-pointer FP32 blocked i-k-j kernel using __restrict pointers for auto-vectorization.
void matmul_fp32_kernel(const float* __restrict a,
                        const float* __restrict b,
                        float* __restrict c,
                        std::size_t M,
                        std::size_t K,
                        std::size_t N,
                        std::size_t block) {
    for (std::size_t ii = 0; ii < M; ii += block) {
        const std::size_t i_end = std::min(ii + block, M);
        for (std::size_t kk = 0; kk < K; kk += block) {
            const std::size_t k_end = std::min(kk + block, K);
            for (std::size_t jj = 0; jj < N; jj += block) {
                const std::size_t j_end = std::min(jj + block, N);

                for (std::size_t i = ii; i < i_end; ++i) {
                    float* __restrict c_row = c + i * N;
                    const float* __restrict a_row = a + i * K;
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const float r = a_row[k];
                        const float* __restrict b_row = b + k * N;
                        for (std::size_t j = jj; j < j_end; ++j) {
                            c_row[j] += r * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

/// Multiplies FP32 matrices A and B using the blocked i-k-j __restrict kernel.
Matrix matmul_fp32(const Matrix& A, const Matrix& B, std::size_t block) {
    assert(A.cols == B.rows);
    assert(block > 0);
    Matrix C(A.rows, B.cols);
    matmul_fp32_kernel(A.data.data(), B.data.data(), C.data.data(),
                       A.rows, A.cols, B.cols, block);
    return C;
}

/// Blocked i-k-j INT8 kernel that accumulates int8 x int8 products safely into int32_t.
void matmul_int8_i32_kernel(const std::int8_t* __restrict a,
                            const std::int8_t* __restrict b,
                            std::int32_t* __restrict c,
                            std::size_t M,
                            std::size_t K,
                            std::size_t N,
                            std::size_t block) {
    for (std::size_t ii = 0; ii < M; ii += block) {
        const std::size_t i_end = std::min(ii + block, M);
        for (std::size_t kk = 0; kk < K; kk += block) {
            const std::size_t k_end = std::min(kk + block, K);
            for (std::size_t jj = 0; jj < N; jj += block) {
                const std::size_t j_end = std::min(jj + block, N);

                for (std::size_t i = ii; i < i_end; ++i) {
                    std::int32_t* __restrict c_row = c + i * N;
                    const std::int8_t* __restrict a_row = a + i * K;
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const std::int32_t r = static_cast<std::int32_t>(a_row[k]);
                        const std::int8_t* __restrict b_row = b + k * N;
                        for (std::size_t j = jj; j < j_end; ++j) {
                            c_row[j] += r * static_cast<std::int32_t>(b_row[j]);
                        }
                    }
                }
            }
        }
    }
}

/// Multiplies quantized INT8 matrices using an int32_t accumulator and rescales by scale_A * scale_B.
Matrix matmul_int8_i32(const QuantizedMatrix& qA, const QuantizedMatrix& qB, std::size_t block) {
    assert(qA.cols == qB.rows);
    assert(block > 0);
    std::vector<std::int32_t> acc(qA.rows * qB.cols, 0);
    matmul_int8_i32_kernel(qA.data.data(), qB.data.data(), acc.data(),
                           qA.rows, qA.cols, qB.cols, block);

    const float out_scale = qA.scale * qB.scale;
    Matrix C(qA.rows, qB.cols);
    for (std::size_t idx = 0; idx < acc.size(); ++idx) {
        C.data[idx] = static_cast<float>(acc[idx]) * out_scale;
    }
    return C;
}

/// Blocked i-k-j INT8 kernel using 16-bit wrapping accumulation to demonstrate int16_t overflow.
void matmul_int8_i16_kernel(const std::int8_t* __restrict a,
                            const std::int8_t* __restrict b,
                            std::uint16_t* __restrict c,
                            std::size_t M,
                            std::size_t K,
                            std::size_t N,
                            std::size_t block) {
    for (std::size_t ii = 0; ii < M; ii += block) {
        const std::size_t i_end = std::min(ii + block, M);
        for (std::size_t kk = 0; kk < K; kk += block) {
            const std::size_t k_end = std::min(kk + block, K);
            for (std::size_t jj = 0; jj < N; jj += block) {
                const std::size_t j_end = std::min(jj + block, N);

                for (std::size_t i = ii; i < i_end; ++i) {
                    std::uint16_t* __restrict c_row = c + i * N;
                    const std::int8_t* __restrict a_row = a + i * K;
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const std::int16_t r = static_cast<std::int16_t>(a_row[k]);
                        const std::int8_t* __restrict b_row = b + k * N;
                        for (std::size_t j = jj; j < j_end; ++j) {
                            const std::int16_t prod = static_cast<std::int16_t>(r * static_cast<std::int16_t>(b_row[j]));
                            c_row[j] = static_cast<std::uint16_t>(c_row[j] + static_cast<std::uint16_t>(prod));
                        }
                    }
                }
            }
        }
    }
}

/// Multiplies quantized INT8 matrices with a 16-bit accumulator (intentionally overflows on large sums).
Matrix matmul_int8_i16(const QuantizedMatrix& qA, const QuantizedMatrix& qB, std::size_t block) {
    assert(qA.cols == qB.rows);
    assert(block > 0);
    // Accumulate in uint16_t for defined 16-bit two's-complement wraparound (no signed overflow UB)
    std::vector<std::uint16_t> acc(qA.rows * qB.cols, 0);
    matmul_int8_i16_kernel(qA.data.data(), qB.data.data(), acc.data(),
                           qA.rows, qA.cols, qB.cols, block);

    const float out_scale = qA.scale * qB.scale;
    Matrix C(qA.rows, qB.cols);
    for (std::size_t idx = 0; idx < acc.size(); ++idx) {
        const std::int16_t wrapped = static_cast<std::int16_t>(acc[idx]);
        C.data[idx] = static_cast<float>(wrapped) * out_scale;
    }
    return C;
}

/// Holds max absolute error, average absolute error, max reference magnitude, and relative error.
struct ErrorStats {
    float max_abs;
    double avg_abs;
    float max_ref;
    double rel_max;
};

/// Computes max, average, and relative error of approx against reference matrix ref.
ErrorStats compute_error(const Matrix& approx, const Matrix& ref) {
    assert(approx.rows == ref.rows && approx.cols == ref.cols);
    float max_abs = 0.0f;
    float max_ref = 0.0f;
    double sum_abs = 0.0;

    for (std::size_t idx = 0; idx < ref.data.size(); ++idx) {
        const float err = std::fabs(approx.data[idx] - ref.data[idx]);
        const float r_abs = std::fabs(ref.data[idx]);
        sum_abs += err;
        if (err > max_abs) max_abs = err;
        if (r_abs > max_ref) max_ref = r_abs;
    }

    const double avg_abs = sum_abs / static_cast<double>(ref.data.size());
    const double rel_max = (max_ref > 0.0f) ? (static_cast<double>(max_abs) / static_cast<double>(max_ref)) : 0.0;
    return {max_abs, avg_abs, max_ref, rel_max};
}

/// Stores speed, memory, and error metrics for one Day 8 CSV row.
struct Day8Row {
    std::string version;
    std::size_t N;
    double seconds;
    double gops;
    std::size_t input_bytes;
    float max_error;
    double avg_error;
    double rel_error;
};

/// Times a matmul callable over multiple runs, computes error against ref, and returns median stats.
Day8Row bench_day8(const std::string& version,
                   const std::function<Matrix()>& fn,
                   const Matrix& ref,
                   std::size_t N,
                   std::size_t input_bytes,
                   int runs = 5) {
    std::vector<double> times_sec;
    times_sec.reserve(runs);
    ErrorStats err{0.0f, 0.0, 0.0f, 0.0};
    volatile float sink = 0.0f;

    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::steady_clock::now();
        Matrix C = fn();
        auto end = std::chrono::steady_clock::now();

        times_sec.push_back(std::chrono::duration<double>(end - start).count());
        sink = C(0, 0);
        if (r == 0) {
            err = compute_error(C, ref);
        }
    }
    (void)sink;

    std::sort(times_sec.begin(), times_sec.end());
    const double median_sec = times_sec[times_sec.size() / 2];
    const double gops = 2.0 * N * N * N / (median_sec * 1e9);
    return {version, N, median_sec, gops, input_bytes, err.max_abs, err.avg_abs, err.rel_max};
}

/// Writes Day 8 speed, memory, and error comparison rows to a CSV file at path.
void write_day8_csv(const std::string& path, const std::vector<Day8Row>& rows) {
    std::ofstream out(path);
    assert(out.is_open());
    out << "version,N,seconds,gflops,input_bytes,max_error,avg_error,rel_error\n";
    out << std::setprecision(6);
    for (const Day8Row& r : rows) {
        out << r.version << ','
            << r.N << ','
            << r.seconds << ','
            << r.gops << ','
            << r.input_bytes << ','
            << r.max_error << ','
            << r.avg_error << ','
            << r.rel_error << '\n';
    }
}

/// Day 8 entry point: benchmarks FP32 vs. INT8 (int32 and int16) at N = 512, 1024 and writes CSV.
int main(int argc, char** argv) {
    const std::string csv_path = (argc > 1) ? argv[1] : "results_day8.csv";
    constexpr std::size_t block = 32;
    const std::vector<std::size_t> sizes = {512, 1024};

    std::vector<Day8Row> rows;
    for (std::size_t N : sizes) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        Matrix A(N, N);
        Matrix B(N, N);
        for (float& x : A.data) x = dist(rng);
        for (float& x : B.data) x = dist(rng);

        const QuantizedMatrix qA = quantize_symmetric(A);
        const QuantizedMatrix qB = quantize_symmetric(B);
        const Matrix ref = matmul_fp32(A, B, block);

        const std::size_t fp32_in_bytes = 2 * N * N * sizeof(float);
        const std::size_t int8_in_bytes = 2 * (N * N * sizeof(std::int8_t) + sizeof(float));

        rows.push_back(bench_day8("fp32_blocked32", [&]() { return matmul_fp32(A, B, block); }, ref, N, fp32_in_bytes));
        rows.push_back(bench_day8("int8_i32_blocked32", [&]() { return matmul_int8_i32(qA, qB, block); }, ref, N, int8_in_bytes));
        rows.push_back(bench_day8("int8_i16_overflow", [&]() { return matmul_int8_i16(qA, qB, block); }, ref, N, int8_in_bytes));
    }

    for (const Day8Row& r : rows) {
        std::cout << r.version << " | N=" << r.N
                  << " | " << r.seconds << " s"
                  << " | " << r.gops << " GOPS"
                  << " | mem=" << (r.input_bytes / (1024.0 * 1024.0)) << " MB"
                  << " | max_err=" << r.max_error
                  << " | avg_err=" << r.avg_error
                  << " | rel_err=" << (100.0 * r.rel_error) << "%\n";
    }

    write_day8_csv(csv_path, rows);
    std::cout << "Wrote " << csv_path << '\n';
    return 0;
}
