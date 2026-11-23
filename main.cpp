#include <cstddef>
#include <vector>
#include <iostream>
#include <cassert>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <fstream>
#include <iomanip>
#include <utility>

/// Row-major 2D matrix of FP32 values stored in a flat 1D vector.
struct Matrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;

    /// Constructs a zero-initialized r x c matrix.
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c) {}

    /// Returns a mutable reference to element (i, j) in row-major layout.
    float& operator()(std::size_t i, std::size_t j) {
        return data[i * cols + j];
    }

    /// Returns a read-only reference to element (i, j) in row-major layout.
    const float& operator()(std::size_t i, std::size_t j) const {
        return data[i * cols + j];
    }
};

/// Fills every element of matrix m with uniform random floats in [-1, 1).
void fill_random(Matrix& m, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : m.data) {
        x = dist(rng);
    }
}

/// Multiplies A and B in naive i-j-k order (inner k: A stride-1, B stride-N, C in register).
Matrix matmul_naive(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

/// Multiplies A and B in j-i-k order (inner k: A stride-1, B stride-N, C in register).
Matrix matmul_jik(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols);
    for (std::size_t j = 0; j < B.cols; ++j) {
        for (std::size_t i = 0; i < A.rows; ++i) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

/// Multiplies A and B in i-k-j order (inner j: B and C stride-1, A in register).
Matrix matmul_ikj(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t k = 0; k < A.cols; ++k) {
            float r = A(i, k);
            for (std::size_t j = 0; j < B.cols; ++j) {
                C(i, j) += r * B(k, j);
            }
        }
    }
    return C;
}

/// Multiplies A and B in k-i-j order (inner j: B and C stride-1, A in register).
Matrix matmul_kij(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols);
    for (std::size_t k = 0; k < A.cols; ++k) {
        for (std::size_t i = 0; i < A.rows; ++i) {
            float r = A(i, k);
            for (std::size_t j = 0; j < B.cols; ++j) {
                C(i, j) += r * B(k, j);
            }
        }
    }
    return C;
}

/// Multiplies A and B in j-k-i order (inner i: A and C stride-N, B in register).
Matrix matmul_jki(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols);
    for (std::size_t j = 0; j < B.cols; ++j) {
        for (std::size_t k = 0; k < A.cols; ++k) {
            float r = B(k, j);
            for (std::size_t i = 0; i < A.rows; ++i) {
                C(i, j) += A(i, k) * r;
            }
        }
    }
    return C;
}

/// Multiplies A and B in k-j-i order (inner i: A and C stride-N, B in register).
Matrix matmul_kji(const Matrix& A, const Matrix& B) {
    assert(A.cols == B.rows);
    Matrix C(A.rows, B.cols);
    for (std::size_t k = 0; k < A.cols; ++k) {
        for (std::size_t j = 0; j < B.cols; ++j) {
            float r = B(k, j);
            for (std::size_t i = 0; i < A.rows; ++i) {
                C(i, j) += A(i, k) * r;
            }
        }
    }
    return C;
}

/// Multiplies A and B using cache-blocked block x block tiles in i-k-j inner order.
Matrix matmul_blocked(const Matrix& A, const Matrix& B, std::size_t block) {
    assert(A.cols == B.rows);
    assert(block > 0);
    Matrix C(A.rows, B.cols);

    for (std::size_t ii = 0; ii < A.rows; ii += block) {
        const std::size_t i_end = std::min(ii + block, A.rows);
        for (std::size_t kk = 0; kk < A.cols; kk += block) {
            const std::size_t k_end = std::min(kk + block, A.cols);
            for (std::size_t jj = 0; jj < B.cols; jj += block) {
                const std::size_t j_end = std::min(jj + block, B.cols);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        float r = A(i, k);
                        for (std::size_t j = jj; j < j_end; ++j) {
                            C(i, j) += r * B(k, j);
                        }
                    }
                }
            }
        }
    }

    return C;
}

/// Multiplies A and B using blocked i-k-j with the innermost j loop hand-unrolled by 4.
Matrix matmul_blocked_unroll4(const Matrix& A, const Matrix& B, std::size_t block) {
    assert(A.cols == B.rows);
    assert(block > 0);
    Matrix C(A.rows, B.cols);

    for (std::size_t ii = 0; ii < A.rows; ii += block) {
        const std::size_t i_end = std::min(ii + block, A.rows);
        for (std::size_t kk = 0; kk < A.cols; kk += block) {
            const std::size_t k_end = std::min(kk + block, A.cols);
            for (std::size_t jj = 0; jj < B.cols; jj += block) {
                const std::size_t j_end = std::min(jj + block, B.cols);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        float r = A(i, k);
                        std::size_t j = jj;
                        for (; j + 3 < j_end; j += 4) {
                            C(i, j)     += r * B(k, j);
                            C(i, j + 1) += r * B(k, j + 1);
                            C(i, j + 2) += r * B(k, j + 2);
                            C(i, j + 3) += r * B(k, j + 3);
                        }
                        for (; j < j_end; ++j) {
                            C(i, j) += r * B(k, j);
                        }
                    }
                }
            }
        }
    }

    return C;
}

/// Raw-pointer blocked i-k-j kernel using __restrict so the compiler skips alias checks and auto-vectorizes.
void matmul_blocked_kernel(const float* __restrict a,
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

/// Multiplies A and B by delegating to the __restrict-qualified blocked kernel.
Matrix matmul_blocked_restrict(const Matrix& A, const Matrix& B, std::size_t block) {
    assert(A.cols == B.rows);
    assert(block > 0);
    Matrix C(A.rows, B.cols);
    matmul_blocked_kernel(A.data.data(), B.data.data(), C.data.data(),
                          A.rows, A.cols, B.cols, block);
    return C;
}

/// Returns the largest element-wise absolute difference |A - B| (or NaN if any diff is NaN).
float max_abs_diff(const Matrix& A, const Matrix& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    float max_diff = 0.0f;
    for (std::size_t idx = 0; idx < A.data.size(); ++idx) {
        float diff = std::fabs(A.data[idx] - B.data[idx]);
        if (std::isnan(diff)) {
            return diff;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

/// Verifies matmul_naive and max_abs_diff against a hand-computed 2x2 example.
void test_2x2() {
    Matrix A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f;
    A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    Matrix B(2, 2);
    B(0, 0) = 5.0f; B(0, 1) = 6.0f;
    B(1, 0) = 7.0f; B(1, 1) = 8.0f;
    Matrix C = matmul_naive(A, B);
    const float expected[2][2] = {
        {19.0f, 22.0f},
        {43.0f, 50.0f}
    };
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            assert(std::fabs(C(i, j) - expected[i][j]) < 1e-5f);
        }
    }
    std::cout << "2x2 test passed:\n"
              << C(0, 0) << ' ' << C(0, 1) << '\n'
              << C(1, 0) << ' ' << C(1, 1) << '\n';

    assert(max_abs_diff(C, C) == 0.0f);
    Matrix perturbed = C;
    perturbed(1, 0) += 0.25f;
    assert(std::fabs(max_abs_diff(C, perturbed) - 0.25f) < 1e-6f);
}

using MatmulFn = std::function<Matrix(const Matrix&, const Matrix&)>;

/// Stores timing, throughput (GFLOP/s), and error metrics for one benchmark run.
struct BenchResult {
    std::string version;
    std::size_t N;
    double seconds;
    double gflops;
    float max_error;
};

/// Runs fn on (A, B) multiple times, verifies against ref, and returns median timing and GFLOP/s.
BenchResult run_benchmark(const std::string& version,
                          MatmulFn fn,
                          const Matrix& A,
                          const Matrix& B,
                          const Matrix& ref,
                          int runs = 5,
                          float tol = 1e-3f) {
    assert(A.rows == A.cols && B.rows == B.cols && A.cols == B.rows);
    const std::size_t N = A.rows;
    std::vector<double> times_sec;
    times_sec.reserve(runs);
    float max_err = 0.0f;
    volatile float sink = 0.0f;
    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::steady_clock::now();
        Matrix C = fn(A, B);
        auto end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(end - start).count();
        times_sec.push_back(sec);
        sink = C(0, 0);
        if (r == 0) {
            max_err = max_abs_diff(C, ref);
            assert(!std::isnan(max_err) && max_err <= tol);
        }
    }
    (void)sink;
    std::sort(times_sec.begin(), times_sec.end());
    double median_sec = times_sec[times_sec.size() / 2];
    double gflops = 2.0 * N * N * N / (median_sec * 1e9);
    return {version, N, median_sec, gflops, max_err};
}

/// Writes all collected benchmark results to a CSV file at path.
void write_csv(const std::string& path, const std::vector<BenchResult>& results) {
    std::ofstream out(path);
    assert(out.is_open());
    out << "version,N,seconds,gflops,max_error\n";
    out << std::setprecision(6);
    for (const BenchResult& r : results) {
        out << r.version << ','
            << r.N << ','
            << r.seconds << ','
            << r.gflops << ','
            << r.max_error << '\n';
    }
}

/// Main benchmark entry point: runs correctness check, benchmarks active versions, and writes CSV.
int main(int argc, char** argv) {
    const std::string csv_path = (argc > 1) ? argv[1] : "results.csv";
    test_2x2();
    const std::vector<std::size_t> sizes = {1024};
    const std::vector<std::pair<std::string, MatmulFn>> versions = {
        {"blocked_32", [](const Matrix& A, const Matrix& B) { return matmul_blocked(A, B, 32); }},
        {"blocked_32_unroll4", [](const Matrix& A, const Matrix& B) { return matmul_blocked_unroll4(A, B, 32); }},
        {"blocked_32_restrict", [](const Matrix& A, const Matrix& B) { return matmul_blocked_restrict(A, B, 32); }},
    };
    std::vector<BenchResult> results;
    for (std::size_t N : sizes) {
        std::mt19937 rng(42);
        Matrix A(N, N);
        Matrix B(N, N);
        fill_random(A, rng);
        fill_random(B, rng);
        Matrix ref = matmul_naive(A, B);
        for (const auto& [name, fn] : versions) {
            BenchResult res = run_benchmark(name, fn, A, B, ref);
            std::cout << res.version << " | N=" << res.N
                      << " | " << res.seconds << " s"
                      << " | " << res.gflops << " GFLOP/s"
                      << " | max_err=" << res.max_error << '\n';
            results.push_back(res);
        }
    }
    write_csv(csv_path, results);
    std::cout << "Wrote " << csv_path << '\n';
    return 0;
}
