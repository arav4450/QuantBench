#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
        const float clamped = std::clamp(rounded, -127.0f, 127.0f);
        qm.data[idx] = static_cast<std::int8_t>(clamped);
    }

    return qm;
}

/// Dequantizes an INT8 matrix back to FP32 by multiplying each element by its scale.
Matrix dequantize_symmetric(const QuantizedMatrix& qm) {
    Matrix m(qm.rows, qm.cols);
    for (std::size_t idx = 0; idx < qm.data.size(); ++idx) {
        m.data[idx] = static_cast<float>(qm.data[idx]) * qm.scale;
    }
    return m;
}

/// Quantizes and dequantizes m, then prints scale, max/avg absolute error, and zero collapse rate.
void report_quant_error(const std::string& label, const Matrix& m) {
    QuantizedMatrix qm = quantize_symmetric(m);
    Matrix m_hat = dequantize_symmetric(qm);

    float max_err = 0.0f;
    double sum_err = 0.0;
    std::size_t zero_count = 0;

    for (std::size_t idx = 0; idx < m.data.size(); ++idx) {
        const float err = std::fabs(m.data[idx] - m_hat.data[idx]);
        sum_err += err;
        if (err > max_err) {
            max_err = err;
        }
        if (qm.data[idx] == 0) {
            ++zero_count;
        }
    }

    const double avg_err = sum_err / static_cast<double>(m.data.size());
    const double zero_pct = 100.0 * static_cast<double>(zero_count) / static_cast<double>(m.data.size());

    std::cout << "\n[" << label << "] (" << m.rows << "x" << m.cols << ")\n"
              << "  scale        = " << qm.scale << '\n'
              << "  max_abs_diff = " << max_err << '\n'
              << "  avg_abs_diff = " << avg_err << '\n'
              << "  quantized==0 = " << zero_count << " / " << m.data.size()
              << " (" << zero_pct << "%)\n";
}

/// Day 7 entry point: verifies 2x2 quantization, compares normal vs. outlier error, and reports memory.
int main() {
    Matrix small(2, 2);
    small(0, 0) = -1.0f; small(0, 1) = -0.25f;
    small(1, 0) =  0.0f; small(1, 1) =  0.5f;

    QuantizedMatrix qsmall = quantize_symmetric(small);
    Matrix small_hat = dequantize_symmetric(qsmall);
    std::cout << "2x2 check (scale=" << qsmall.scale << "):\n";
    for (std::size_t idx = 0; idx < small.data.size(); ++idx) {
        std::cout << "  orig=" << small.data[idx]
                  << " -> q=" << static_cast<int>(qsmall.data[idx])
                  << " -> x_hat=" << small_hat.data[idx]
                  << " (err=" << std::fabs(small.data[idx] - small_hat.data[idx]) << ")\n";
    }

    constexpr std::size_t N = 1024;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Matrix normal(N, N);
    for (float& x : normal.data) {
        x = dist(rng);
    }

    Matrix with_outlier = normal;
    with_outlier(0, 0) = 100.0f;

    report_quant_error("Normal [-1, 1)", normal);
    report_quant_error("Single outlier (100.0 at (0,0))", with_outlier);

    const std::size_t fp32_bytes = N * N * sizeof(float);
    const std::size_t int8_bytes = N * N * sizeof(std::int8_t);
    std::cout << "\nMemory (" << N << "x" << N << "):\n"
              << "  FP32: " << fp32_bytes << " bytes (" << (fp32_bytes / (1024.0 * 1024.0)) << " MB)\n"
              << "  INT8: " << int8_bytes << " bytes + 4 B scale ("
              << (int8_bytes / (1024.0 * 1024.0)) << " MB) -> "
              << (static_cast<double>(fp32_bytes) / static_cast<double>(int8_bytes + sizeof(float)))
              << "x smaller\n";
    return 0;
}
