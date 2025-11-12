#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <complex>
#include <limits>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==================== Utility: Dense Grid ====================//
//
// Build a dense frequency grid W on (0, π) with n_w points (no endpoints).
// The k-th frequency is ω_k = k * π / (n_w + 1), for k = 1..n_w.
//
static std::vector<double> make_dense_grid(int n_w) {
    std::vector<double> w;
    w.reserve(n_w);
    const double step = M_PI / (double)(n_w + 1);
    for (int k = 1; k <= n_w; ++k) w.push_back(step * (double)k);
    return w;
}

//==================== Utility: Quantile ====================//
//
// Quantile of a *sorted* vector with linear interpolation (R-like).
// alpha ∈ [0,1]. Ends are clamped.
//
static double quantile_interp_sorted(const std::vector<double>& sorted, double alpha) {
    const int N = (int)sorted.size();
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();
    if (alpha <= 0.0) return sorted.front();
    if (alpha >= 1.0) return sorted.back();
    const double pos = alpha * (N - 1);
    const int lo = (int)std::floor(pos);
    const int hi = (int)std::ceil(pos);
    const double w = pos - lo;
    if (hi == lo) return sorted[lo];
    return (1.0 - w) * sorted[lo] + w * sorted[hi];
}

//==================== Utility: Neighborhood Pruning ====================//
//
// Remove every index in keep_1b that is within ±d_rad (in radians) of ANY
// index in selected_1b. Indices are *1-based* for convenience when used
// against the dense frequency grid W.
//
static std::vector<int> prune_neighborhood(const std::vector<int>& keep_1b,
                                           const std::vector<int>& selected_1b,
                                           const std::vector<double>& w,
                                           double d_rad) {
    std::vector<int> out; out.reserve(keep_1b.size());
    for (int j1 : keep_1b) {
        const double wi = w[j1 - 1];
        bool bad = false;
        for (int s1 : selected_1b) {
            if (std::fabs(wi - w[s1 - 1]) <= d_rad) { bad = true; break; }
        }
        if (!bad) out.push_back(j1);
    }
    return out;
}

//==================== Utility: Nearest Index on Grid ====================//
//
// Return the 0-based index of the element in sorted vector w that is closest
// to "target" (ties broken by lower distance).
//
static int nearest_index_in_w(const std::vector<double>& w, double target) {
    if (w.empty()) return -1;
    auto it = std::lower_bound(w.begin(), w.end(), target);
    if (it == w.begin()) return 0;
    if (it == w.end())   return (int)w.size() - 1;
    const int hi = (int)std::distance(w.begin(), it);
    const int lo = hi - 1;
    return (std::fabs(w[lo] - target) <= std::fabs(w[hi] - target)) ? lo : hi;
}

//==================== Utility: Coarse Grid Mapping ====================//
//
// Map an *explicit number* of equally spaced target frequencies on (0, π)
// to their nearest indices on the dense grid. Return *1-based* indices.
// If coarse_target_count <= 0, use floor(sqrt(n)) as a default.
//
static std::vector<int> build_coarse_indices(int n,
                                             const std::vector<double>& w_dense,
                                             int coarse_target_count /*<=0 -> sqrt(n)*/) {
    const int F_dense = (int)w_dense.size();
    if (F_dense <= 0) return {};

    int F0 = (coarse_target_count > 0)
           ? coarse_target_count
           : std::max(1, (int)std::floor(std::sqrt((double)n)));

    // Clamp to [1, F_dense] for safety.
    F0 = std::max(1, std::min(F0, F_dense));

    const double dwc = M_PI / (double)F0; // target spacing
    std::vector<int> idx; idx.reserve(F0);
    int last = -1;
    for (int k = 1; k <= F0; ++k) {
        const double omega = k * dwc;
        const int j0 = nearest_index_in_w(w_dense, omega); // 0-based
        if (j0 < 0) continue;
        if (j0 == last) continue;                           // deduplicate if needed
        last = j0;
        idx.push_back(j0 + 1);                              // store as 1-based
    }
    return idx;
}

//==================== DPPT: Compute F(W) on Dense Grid ====================//
static std::vector<double> compute_F_W_parallel(const std::vector<double>& ts,
                                                const std::vector<double>& w) {
    const int n = (int)ts.size();
    const int F = (int)w.size();
    std::vector<double> T(F, 0.0);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < F; ++j) {
        const double omega = w[j];
        const std::complex<double> e = std::exp(std::complex<double>(0.0, omega));
        std::complex<double> z(1.0, 0.0);
        std::complex<double> s(0.0, 0.0);
        double mx = 0.0;

        for (int i = 0; i < n; ++i) {
            s += ts[i] * z;           // accumulate prefix sum in rotating frame
            const double v = std::abs(s);
            if (v > mx) mx = v;
            z *= e;                   // next e^{i ω}
        }
        T[j] = mx;
    }
    return T;
}

//==================== OBMB: Critical Value on Sub-Grid ====================//
static double compute_crit_on_grid(const std::vector<double>& ts,
                                   const std::vector<double>& w_sub,
                                   int n, int m, int K, double alpha,
                                   bool pregen_G = true) {
    const int Fsub = (int)w_sub.size();
    if (Fsub == 0) return 0.0;
    const int rows = n - m + 1;
    if (rows <= 0) return 0.0;

    // Precompute Y_j[r] for each sub-grid frequency using a sliding window.
    std::vector<std::vector<std::complex<double>>> Y(Fsub);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < Fsub; ++j) {
        const double omega = w_sub[j];
        const std::complex<double> e   = std::exp(std::complex<double>(0.0, omega));
        const std::complex<double> e_m = std::exp(std::complex<double>(0.0, omega * (double)m));

        std::vector<std::complex<double>> y(rows);
        std::complex<double> z(1.0, 0.0);
        std::complex<double> S(0.0, 0.0);

        // First window sum: r = 0
        for (int i = 0; i < m; ++i) { S += ts[i] * z; z *= e; }
        y[0] = S;

        // Slide the window: leave ts[r-1], enter ts[r+m-1]
        std::complex<double> z_leave(1.0, 0.0);
        std::complex<double> z_enter = e_m;
        for (int r = 1; r < rows; ++r) {
            S += ts[r + m - 1] * z_enter;
            S -= ts[r - 1] * z_leave;
            y[r] = S;
            z_enter *= e;
            z_leave *= e;
        }
        Y[j].swap(y);
    }

    // Optionally pre-generate multipliers for reproducibility/perf.
    std::vector<double> row_max(K, 0.0);
    std::vector<double> G_pool;
    if (pregen_G) {
        std::mt19937 gen(0xC001D00Du);
        std::normal_distribution<double> N01(0.0, 1.0);
        G_pool.resize((size_t)K * (size_t)rows);
        for (int k = 0; k < K; ++k) {
            double* gk = G_pool.data() + (size_t)k * (size_t)rows;
            for (int r = 0; r < rows; ++r) gk[r] = N01(gen);
        }
    }

    // Bootstrap loop.
    #pragma omp parallel
    {
        std::mt19937 gen_loc(0xBADC0FFEu ^
            (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()
        #ifdef _OPENMP
            ^ (unsigned)omp_get_thread_num() * 0x9e3779b9U
        #endif
        );
        std::normal_distribution<double> N01(0.0, 1.0);
        std::vector<double> G(rows);

        #pragma omp for schedule(static)
        for (int k = 0; k < K; ++k) {
            const double* gk = nullptr;
            if (pregen_G) {
                gk = G_pool.data() + (size_t)k * (size_t)rows;
            } else {
                for (int r = 0; r < rows; ++r) G[r] = N01(gen_loc);
                gk = G.data();
            }

            double mx_over_freq = 0.0;

            for (int j = 0; j < Fsub; ++j) {
                const auto& y = Y[j];
                std::complex<double> s(0.0, 0.0);
                double mx = 0.0;
                for (int r = 0; r < rows; ++r) {
                    s += y[r] * gk[r];
                    const double v = std::abs(s);
                    if (v > mx) mx = v;
                }
                if (mx > mx_over_freq) mx_over_freq = mx;
            }
            row_max[k] = mx_over_freq;
        }
    }

    std::sort(row_max.begin(), row_max.end());
    const double q = quantile_interp_sorted(row_max, 1.0 - alpha);
    return q / std::sqrt((double)m * (double)(n - m));
}

//==================== Stage 1 ====================//
//
// Implements Stage-1 (DPPT + OBMB) with an engineering acceleration:
//   • A *coarse* grid nominates centers (default ≈ √n).
//   • For each center, build a *refined* local sub-grid around it:
//       - half-width in *index units* (default ≈ √n),
//       - cap the sub-grid length (default 1000) via uniform downsampling,
//         always keeping the local T(ω) peak.
//   • Test the peak against an OBMB critical value computed on that refined sub-grid.
//   • If significant, accept and globally remove a neighborhood of width
//     d_rad = log(m)/(4√m) in *radians*; otherwise count a "miss".
//   • Early stop after 'max_consecutive_fails' consecutive misses.
//   • Return the list of accepted angular frequencies (radians/sample).
//
// Extended signature with tunables; see wrapper below for the 4-arg form.
//
std::vector<double>
get_est_cpp(const std::vector<double> &time_series,
            int n_w,
            double alpha,
            int K,
            int coarse_target_count /*<=0 -> sqrt(n)*/ = -1,
            int max_consecutive_fails = 4,
            int refined_grid_len = 1000,
            int refined_radius /*<=0 -> sqrt(n)*/ = -1,
            int nf_max = 10) {
    const int n = (int)time_series.size();
    if (n < 3 || n_w <= 0) return {};

    // Stage-1 defaults: m ≈ n^{1/3}; global neighborhood width d_rad.
    const int m = std::max(1, (int)std::floor(std::pow((double)n, 1.0/3.0)));
    const double d_rad = std::log((double)m) / (4.0 * std::sqrt((double)m));

    // 1) Dense grid and DPPT on the grid.
    const std::vector<double> w_dense = make_dense_grid(n_w);
    const int F = (int)w_dense.size();
    if (F == 0) return {};
    const std::vector<double> T = compute_F_W_parallel(time_series, w_dense);

    // 2) Coarse centers: map targets to dense grid, then sort by T descending.
    std::vector<int> coarse_1b = build_coarse_indices(n, w_dense, coarse_target_count); // 1-based
    if (coarse_1b.empty()) return {};
    std::vector<std::pair<double,int>> coarse_sorted; // (T value, 1-based index)
    coarse_sorted.reserve(coarse_1b.size());
    for (int j1 : coarse_1b) coarse_sorted.emplace_back(T[j1 - 1], j1);
    std::sort(coarse_sorted.begin(), coarse_sorted.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    // 3) Global mask over W (1-based); 1=removed.
    std::vector<char> masked(F + 1, 0);

    // 4) Selection loop with early-stopping on consecutive misses.
    const int half_width = (refined_radius > 0)
                         ? refined_radius
                         : std::max(1, (int)std::floor(std::sqrt((double)n)));
    const int cap = (refined_grid_len > 0) ? refined_grid_len : 1000;

    std::vector<int> chosen; chosen.reserve(nf_max); // store 1-based indices
    int accepted = 0, misses = 0;

    for (const auto& kv : coarse_sorted) {
        if (accepted >= nf_max) break;
        if (misses >= max_consecutive_fails) break;

        const int center1b = kv.second;
        if (masked[center1b]) continue;

        // Build refined window [L..R] around center and drop masked points.
        const int L = std::max(1, center1b - half_width);
        const int R = std::min(F, center1b + half_width);

        std::vector<int> ifr_full; ifr_full.reserve(R - L + 1);
        for (int j1 = L; j1 <= R; ++j1) if (!masked[j1]) ifr_full.push_back(j1);
        if (ifr_full.empty()) continue;

        // Find local peak on the *full* refined window (before downsampling).
        double peak_val = -std::numeric_limits<double>::infinity();
        int    peak1b   = -1;
        for (int j1 : ifr_full) {
            const double v = T[j1 - 1];
            if (v > peak_val) { peak_val = v; peak1b = j1; }
        }
        if (peak1b < 0) continue;

        // Uniform downsampling to 'cap' points, always keep the local peak.
        std::vector<int> ifr = ifr_full;
        if ((int)ifr_full.size() > cap) {
            std::vector<int> tmp; tmp.reserve(cap + 2);
            const double step = (double)ifr_full.size() / (double)cap;
            int last = -1;
            for (int t = 0; t < cap; ++t) {
                int pos = (int)std::floor(t * step);
                if (pos >= (int)ifr_full.size()) pos = (int)ifr_full.size() - 1;
                const int j1 = ifr_full[pos];
                if (j1 != last) { tmp.push_back(j1); last = j1; }
            }
            if (tmp.empty() || tmp.back() != ifr_full.back()) tmp.push_back(ifr_full.back());
            // Force-keep the peak.
            tmp.push_back(peak1b);
            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
            ifr.swap(tmp);
        }

        // Test statistic at the local peak (normalize by sqrt(n)).
        const double tst = peak_val / std::sqrt((double)n);

        // OBMB critical value on the current refined sub-grid.
        std::vector<double> w_ifr; w_ifr.reserve(ifr.size());
        for (int j1 : ifr) w_ifr.push_back(w_dense[j1 - 1]);
        const double crit = compute_crit_on_grid(time_series, w_ifr, n, m, K, alpha, true);

        if (tst <= crit) {
            // Not significant -> record a miss and continue.
            ++misses;
            continue;
        }

        // Significant -> accept, reset misses, and globally remove a log(m)/(4√m) neighborhood.
        chosen.push_back(peak1b);
        ++accepted;
        misses = 0;

        const double ws = w_dense[peak1b - 1];
        for (int j1 = 1; j1 <= F; ++j1) {
            if (!masked[j1] && std::fabs(w_dense[j1 - 1] - ws) <= d_rad) masked[j1] = 1;
        }
    }

    // Return selected angular frequencies (radians/sample) in ascending order.
    std::sort(chosen.begin(), chosen.end());
    std::vector<double> omegas; omegas.reserve(chosen.size());
    for (int j1 : chosen) omegas.push_back(w_dense[j1 - 1]);
    return omegas;
}
// ==================== Wrapper: 4-para get_est_cpp ====================//
//It uses the defaults:
//   - coarse_target_count ≈ sqrt(n)
//   - max_consecutive_fails = 4
//   - refined_grid_len = 1000
//   - refined_radius ≈ sqrt(n)
//   - nf_max = 10
//
std::vector<double>
get_est_cpp(const std::vector<double> &time_series,
            int n_w,
            double alpha,
            int K) {
    return get_est_cpp(time_series, n_w, alpha, K,
                       /*coarse_target_count=*/-1,
                       /*max_consecutive_fails=*/4,
                       /*refined_grid_len=*/1000,
                       /*refined_radius=*/-1,
                       /*nf_max=*/10);
}

