# QuantBench

Matrix multiplication benchmarking and optimization lab in C++17.

## Build & Run

One command to rebuild (`-O2`) and regenerate `results.csv` across `N = 128, 256, 512, 1024`:

```bash
cmake --build build && ./build/quantbench
```

Optional `-O0` build for comparison:

```bash
cmake -S . -B build_O0 -DOPT_FLAG=-O0 && cmake --build build_O0 && ./build_O0/quantbench results_O0.csv
```

## Results

All timings are the median of 5 runs (`std::chrono::steady_clock`, fixed seed `42`).

### Day 2: Naive Baseline (`-O0` vs `-O2`)

| Version | N | Median (s) | GFLOP/s | Max Error | Speedup vs `-O0` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `naive (-O0)` | 128 | 0.022604 | 0.19 | 0 | 1.0× |
| `naive (-O0)` | 256 | 0.191092 | 0.18 | 0 | 1.0× |
| `naive (-O0)` | 512 | 1.386560 | 0.19 | 0 | 1.0× |
| `naive (-O0)` | 1024 | 12.345300 | 0.17 | 0 | 1.0× |
| `naive (-O2)` | 128 | 0.002498 | 1.68 | 0 | 9.1× |
| `naive (-O2)` | 256 | 0.023404 | 1.43 | 0 | 8.2× |
| `naive (-O2)` | 512 | 0.164175 | 1.64 | 0 | 8.4× |
| `naive (-O2)` | 1024 | 4.817640 | 0.45 | 0 | 2.6× |

**Takeaway:** `-O2` gives an `8–9×` speedup while matrices fit in cache by inlining accessors and keeping accumulators in registers, but drops to `2.6×` at `N = 1024` where strided `B(k, j)` cache misses become the bottleneck.


### Day 3: Loop Ordering & Cache Locality (`-O2`)

| Order | Inner Loop | Stride-1 Access | Row-Jump (Stride `N`) | `N = 512` (s) | `N = 512` GFLOP/s | `N = 1024` (s) | `N = 1024` GFLOP/s | Speedup vs `ijk` (`1024`) |
| :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`ikj`** | `j` | `B`, `C` | none (`A` in reg) | 0.067164 | 4.00 | 0.518625 | **4.14** | **7.4×** |
| **`kij`** | `j` | `B`, `C` | none (`A` in reg) | 0.062911 | **4.27** | 0.542128 | 3.96 | 7.1× |
| `jik` | `k` | `A` | `B` (`C` in reg) | 0.167416 | 1.60 | 3.829330 | 0.56 | 1.0× |
| `ijk` | `k` | `A` | `B` (`C` in reg) | 0.170238 | 1.58 | 3.848000 | 0.56 | 1.0× |
| `kji` | `i` | none | `A`, `C` (`B` in reg) | 0.536791 | 0.50 | 8.943700 | 0.24 | 0.43× |
| `jki` | `i` | none | `A`, `C` (`B` in reg) | 0.549940 | 0.49 | 9.314280 | 0.23 | 0.41× |

**Why `ikj` wins:** Keeping `j` innermost holds `A(i, k)` in a register while streaming both `B(k, j)` and `C(i, j)` with stride 1 (using every float of each 64-byte cache line and keeping one row of `C` hot in L1), eliminating the 4 KB stride cache-conflict cliff at `N = 1024`.

### Day 4: Cache Blocking / Tiling (`N = 1024`, `-O2`)

CPU caches (`lscpu -C`): **L1d = 48 KB** (12-way, 64 sets × 64 B = 4 KB/way), **L2 = 2 MB** (16-way), **L3 = 24 MB**.

| Version | 3-Tile Working Set (`3×B²×4`) | Median (s) | GFLOP/s | Max Error | vs `ikj` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`ikj` (unblocked)** | — | 0.493055 | **4.36** | 0 | **1.00×** |
| `blocked_32` | 12 KB | 0.588658 | 3.65 | 0 | 0.84× |
| `blocked_128` | 192 KB | 0.589565 | 3.64 | 0 | 0.84× |
| `blocked_64` | 48 KB | 0.656992 | 3.27 | 0 | 0.75× |
| `blocked_16` | 3 KB | 0.706842 | 3.04 | 0 | 0.70× |

**Takeaway:** Among blocked sizes, `B = 32` (`12 KB`, fits L1d with headroom) and `B = 128` (`192 KB`, fits `2 MB` L2 with lower loop overhead) were fastest, while `B = 16` suffered from short inner-loop overhead and `B = 64` suffered from L1d set-associativity conflict misses (at `N = 1024`, rows are exact `4 KB` apart, mapping every row of a tile to the same 12-way L1d sets). Unblocked `ikj` still won at `N = 1024` because all three matrices (`12 MB`) fit inside the `24 MB` L3 cache where long `1024`-float streams favor hardware prefetching.

### Day 5: Compiler Flags, Loop Unrolling & `__restrict` (`N = 1024`)

#### Flag sweep on `blocked_32`

| Flags | Inner-Loop Instructions | Median (s) | GFLOP/s | Max Error | Speedup vs `-O0` |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `-O0` | Scalar stack ops | 11.716800 | 0.18 | 0 | 1.0× |
| `-O2` | `mulss`, `addss` (1 float/op) | 0.585614 | 3.67 | 0 | 20.0× |
| `-O3` | `mulps`, `addps` (SSE 4 floats/op) | 0.225626 | 9.52 | 0 | 51.9× |
| **`-O3 -march=native`** | `vbroadcastss`, **`vfmadd213ps`** (AVX2 8 floats/op, FMA) | 0.220547 | **9.74** | `2.29e-5` | **53.1×** |

#### Unroll×4 & `__restrict` comparison (`N = 1024`)

| Version | Flags | Median (s) | GFLOP/s | Max Error |
| :--- | :--- | ---: | ---: | ---: |
| `blocked_32` | `-O2` | 0.605657 | 3.55 | 0 |
| `blocked_32_unroll4` | `-O2` | 0.491788 | 4.37 | 0 |
| **`blocked_32_restrict`** | `-O2` | 0.175482 | **12.24** | 0 |
| `blocked_32` | `-O3 -march=native` | 0.243191 | 8.83 | `2.29e-5` |
| `blocked_32_unroll4` | `-O3 -march=native` | 0.220063 | 9.76 | `2.29e-5` |
| **`blocked_32_restrict`** | **`-O3 -march=native`** | **0.167556** | **12.82** | `2.29e-5` |

**Flags going forward:** **`-O3 -march=native`** paired with **`__restrict`** pointers. `-O3` enables aggressive loop vectorization/unroll-and-jam, `-march=native` unlocks 256-bit AVX2 + FMA (`vfmadd213ps` / `vfmadd231ps`), and `__restrict` removes aliasing ambiguity so GCC eliminates runtime overlap checks and unroll-and-jams two `k` iterations per `C` load/store—reaching **`12.82 GFLOP/s`** (`23×` faster than Day 2's `-O2` naive baseline).

### Day 6: FP32 & BF16 Bit Representation (`day6.cpp`)

#### Bit layout example: `0.1f` (FP32 vs. BF16 truncation)

```text
FP32 (32 bits): [ 0 | 01111011 | 1001100 1100110011001101 ]  ->  0.100000001
BF16 (16 bits): [ 0 | 01111011 | 1001100 0000000000000000 ]  ->  0.099609375
                  ^   ^^^^^^^^   ^^^^^^^ ^^^^^^^^^^^^^^^^
                 sign  exp (8)   mant (7)   dropped (16)
                       raw=123, E=-4
```

#### Measured BF16 round-trip error (`1024×1024` random matrix in `[-1, 1)`, seed `42`)

| Metric | Measured | Theoretical (`[-1, 1)`) |
| :--- | ---: | ---: |
| `max_abs_diff` | `0.00390619` | `1/256 = 0.00390625` |
| `avg_abs_diff` | `0.00130129` | `1/768 = 0.00130208` |

**Why AI chips like BF16:** BF16 keeps all 8 exponent bits of FP32, giving it the exact same dynamic range ($\sim 10^{-38}$ to $3.4 \times 10^{38}$) so neural-network gradients don't overflow or underflow the way they do in IEEE FP16. Dropping the bottom 16 mantissa bits halves memory bandwidth and shrinks hardware multipliers dramatically—while neural networks tolerate ~`0.13%` average rounding noise cleanly—and makes FP32↔BF16 conversion a trivial 16-bit shift.

### Day 7: Symmetric Per-Tensor INT8 Quantization (`day7.cpp`, `1024×1024`, seed `42`)

#### Round-trip quantization error (normal vs. single outlier)

| Case | `scale` | `max_abs_diff` | `avg_abs_diff` | Values Quantized to `0` |
| :--- | ---: | ---: | ---: | ---: |
| Normal `[-1, 1)` | `0.007874` | `0.003937` | `0.001967` | `0.39%` (`4,094`) |
| Single outlier (`100.0` at `(0, 0)`) | `0.787402` | `0.393700` | `0.177597` | `39.41%` (`413,246`) |

#### Memory footprint (`N = 1024`)

| Format | Formula | Bytes | Size | Compression |
| :--- | :--- | ---: | ---: | ---: |
| FP32 | `N × N × 4` | `4,194,304` | `4.0 MB` | `1.0×` |
| INT8 (per-tensor) | `N × N × 1 + 4` | `1,048,580` | `1.0 MB` | **`4.0×`** |

**Takeaway:** Symmetric per-tensor INT8 shrinks matrix storage by **`4×`** with tiny error on well-behaved distributions (`max_abs_diff = 0.00394`, matching 2-byte BF16's worst-case error on `[-1, 1)`). However, a single `100.0` outlier inflates the shared scale by `100×`, multiplying error by `90–100×` and collapsing `39.4%` of all values to `0` (while leaving 252 of 255 INT8 buckets unused)—which is why real LLM quantization uses per-channel or group-wise scales.

### Day 8: INT8 Matrix Multiply (`day8.cpp`, `-O3 -march=native`, `block = 32`)

Output CSV: `results_day8.csv` (`./build/day8`).

| Version | N | Median (s) | GOPS | Input Memory (`A + B`) | Max Error | Avg Error | Rel Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32_blocked32` | 512 | 0.022797 | 11.78 | 2.00 MB | 0 | 0 | 0% |
| `int8_i32_blocked32` | 512 | 0.025302 | 10.61 | **0.50 MB (`4×`)** | 0.2030 | 0.0334 | **0.59%** |
| `int8_i16_overflow` | 512 | 0.012970 | 20.70 | 0.50 MB | 36.4803 | 5.9485 | 105.89% *(overflow)* |
| `fp32_blocked32` | 1024 | 0.167952 | **12.79** | 8.00 MB | 0 | 0 | 0% |
| `int8_i32_blocked32` | 1024 | 0.170815 | 12.57 | **2.00 MB (`4×`)** | 0.3045 | 0.0473 | **0.59%** |
| `int8_i16_overflow` | 1024 | 0.104659 | 20.52 | 2.00 MB | 52.8703 | 8.4579 | 102.19% *(overflow)* |

**Takeaways:**
- **Why `int32_t` is required over `int16_t`:** Each `int8 × int8` product is at most $127^2 = 16{,}129$ (fits in `int16_t`), but summing just 3 large products overflows `32,767`. Even though `int16_t` packs 2× more lanes per AVX2 register (`20.5 GOPS`), 16-bit wraparound destroys the output (`>102%` relative error). `int32_t` safely sums up to `133,144` max products, keeping relative error at `0.59%` while cutting input memory by `4×`.
- **Why scalar-C++ INT8 (`12.57 GOPS`) didn't beat FP32 (`12.79 GOPS`):** Widening `int8` → `int32` still fits only 8 lanes per 256-bit `ymm` register (same width as 8 `float`s) without a single fused FMA instruction; getting a 2×–4× INT8 compute speedup requires explicit SIMD dot-product intrinsics (`vpmaddubsw`/`vpmaddwd`, AVX-VNNI `vpdpbusd`) or dedicated INT8 hardware units.
