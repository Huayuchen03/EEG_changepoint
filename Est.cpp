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

//==================== Utility Functions ====================//

// Create a dense grid of frequencies between (0, π).
// ω_k = k * π / (n_w + 1), for k = 1..n_w.
static std::vector<double> make_dense_grid(int n_w) {
    std::vector<double> w;
    w.reserve(n_w);
    const double step = M_PI / (double)(n_w + 1);
    for (int k = 1; k <= n_w; ++k) w.push_back(step * (double)k);
    return w;
}

// Compute a quantile of a sorted array using linear interpolation.
static double quantile_interp_sorted(const std::vector<double>& sorted, double alpha) {
    int N = (int)sorted.size();
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();
    if (alpha <= 0.0) return sorted.front();
    if (alpha >= 1.0) return sorted.back();
    double pos = alpha * (N - 1);
    int lo = (int)std::floor(pos);
    int hi = (int)std::ceil(pos);
    double w = pos - lo;
    if (hi == lo) return sorted[lo];
    return (1.0 - w) * sorted[lo] + w * sorted[hi];
}

// Remove all elements from 'keep_1b' that are too close (≤ d_rad)
// to any element in 'selected_1b'. Both use 1-based indices.
static std::vector<int> prune_neighborhood(const std::vector<int>& keep_1b,
                                           const std::vector<int>& selected_1b,
                                           const std::vector<double>& w,
                                           double d_rad) {
    std::vector<int> out;
    out.reserve(keep_1b.size());
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

//==================== Function to Compute F(W) ====================//
// For each frequency ω, compute T(ω) = max_i |∑_{k=0..i} x_k e^{iωk}|
// Returns a vector of T values for each frequency in w.
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
            s += ts[i] * z;
            const double v = std::abs(s);
            if (v > mx) mx = v;
            z *= e;
        }
        T[j] = mx;
    }
    return T;
}

//==================== Compute Bootstrap Critical Value ====================//
// Calculate the critical threshold used for testing frequency significance.
// Performs bootstrap replications using Gaussian multipliers.
static double compute_crit_on_grid(const std::vector<double>& ts,
                                   const std::vector<double>& w_sub,
                                   int n, int m, int K, double alpha,
                                   bool pregen_G = true) {
    const int Fsub = (int)w_sub.size();
    if (Fsub == 0) return 0.0;
    const int rows = n - m + 1;
    if (rows <= 0) return 0.0;

    // Precompute moving window sums for each frequency.
    std::vector<std::vector<std::complex<double>>> Y(Fsub);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < Fsub; ++j) {
        const double omega = w_sub[j];
        const std::complex<double> e = std::exp(std::complex<double>(0.0, omega));
        const std::complex<double> e_m = std::exp(std::complex<double>(0.0, omega * (double)m));

        std::vector<std::complex<double>> y(rows);
        std::complex<double> z(1.0, 0.0);
        std::complex<double> S(0.0, 0.0);
        for (int i = 0; i < m; ++i) { S += ts[i] * z; z *= e; }
        y[0] = S;

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

    // Pre-generate Gaussian random numbers (if requested).
    std::vector<double> row_max(K, 0.0);
    std::vector<double> G_pool; G_pool.reserve((size_t)K * (size_t)rows);
    if (pregen_G) {
        std::mt19937 gen(0xC001D00Du); //seed
        std::normal_distribution<double> N01(0.0, 1.0);
        G_pool.resize((size_t)K * (size_t)rows);
        for (int k = 0; k < K; ++k) {
            double* gk = G_pool.data() + (size_t)k * (size_t)rows;
            for (int r = 0; r < rows; ++r) gk[r] = N01(gen);
        }
    }

    // Perform bootstrap simulations.
    #pragma omp parallel
    {
        std::mt19937 gen_loc(0xBADC0FFEu ^
            (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()
        #ifdef _OPENMP
            ^ (unsigned)omp_get_thread_num()*0x9e3779b9U
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

    // Return the (1 - alpha) quantile of bootstrap maxima.
    std::sort(row_max.begin(), row_max.end());
    const double q = quantile_interp_sorted(row_max, 1.0 - alpha);
    return q / std::sqrt((double)m * (double)(n - m));
}

//==================== Stage 1: Frequency Selection ====================//
// Implements the iterative frequency detection process:
// - Compute T(ω) on all frequencies
// - Iteratively test the strongest frequency
// - Keep it if significant and remove nearby frequencies
std::vector<double>
get_est_cpp(const std::vector<double> &time_series,
            int n_w,
            double alpha,
            int K) {
    const int n = (int)time_series.size();
    if (n < 3 || n_w <= 0) return {};

    // Define parameters for window size and neighborhood distance.
    const int m = std::max(1, (int)std::floor(std::pow((double)n, 1.0/3.0)));
    const double d_rad = std::log((double)m) / (4.0 * std::sqrt((double)m));

    // Create a dense frequency grid.
    const std::vector<double> w_dense = make_dense_grid(n_w);
    const int F = (int)w_dense.size();

    // Compute F(W) values for all frequencies.
    const std::vector<double> T = compute_F_W_parallel(time_series, w_dense);

    // Initialize all frequencies as active (1-based indices).
    std::vector<int> active; active.reserve(F);
    for (int j1 = 1; j1 <= F; ++j1) active.push_back(j1);
    std::vector<int> selected_1b; selected_1b.reserve(8);

    // Iteratively select significant frequencies.
    while (!active.empty()) {
        // Build the current subset of active frequencies.
        std::vector<double> w_k; w_k.reserve(active.size());
        for (int j1 : active) w_k.push_back(w_dense[j1 - 1]);

        // Compute critical value for the current subset.
        const double crit = compute_crit_on_grid(time_series, w_k, n, m, K, alpha, true);

        // Find the maximum T among active frequencies.
        double mt = -std::numeric_limits<double>::infinity();
        int best_j1 = -1;
        #pragma omp parallel
        {
            double lmt = -std::numeric_limits<double>::infinity();
            int lb = -1;
            #pragma omp for nowait schedule(static)
            for (int idx = 0; idx < (int)active.size(); ++idx) {
                const int j1 = active[idx];
                const double v = T[j1 - 1];
                if (v > lmt) { lmt = v; lb = j1; }
            }
            #pragma omp critical
            { if (lmt > mt) { mt = lmt; best_j1 = lb; } }
        }
        if (best_j1 < 0) break;

        const double test_stat = mt / std::sqrt((double)n);
        if (test_stat <= crit) break;  // Stop if not significant.

        // Accept this frequency and remove nearby ones.
        selected_1b.push_back(best_j1);
        std::vector<int> tmp = { best_j1 };
        active = prune_neighborhood(active, tmp, w_dense, d_rad);
    }

    // Return selected frequencies in ascending order.
    std::sort(selected_1b.begin(), selected_1b.end());
    std::vector<double> omegas; omegas.reserve(selected_1b.size());
    for (int j1 : selected_1b) omegas.push_back(w_dense[j1 - 1]);
    return omegas;
}