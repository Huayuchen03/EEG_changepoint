// ---- helpers: dense grid on (0, π) with n_w points (no endpoints) ----
static std::vector<double> make_dense_grid(int n_w) {
    std::vector<double> w;
    w.reserve(n_w);
    const double step = M_PI / (double)(n_w + 1); // ω_k = k * π/(n_w+1), k=1..n_w
    for (int k = 1; k <= n_w; ++k)
        w.push_back(step * (double)k);
    return w;
}

// ---- cum-sum max-abs of a complex array ----
static double cs_s1(const std::vector<std::complex<double>>& x) {
    const int n = (int)x.size();
    std::complex<double> s(0.0, 0.0);
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        s += x[i];
        double v = std::abs(s);
        if (v > mx) mx = v;
    }
    return mx;
}

// ---- F(W): for each ω, compute cs_s1(x_i * e^{i ω i}) ----
static std::vector<double> compute_F_W_parallel(const std::vector<double>& ts, const std::vector<double>& w) {
    const int n = (int)ts.size();
    const int F = (int)w.size();
    std::vector<double> T(F, 0.0);
#pragma omp parallel for schedule(static)
    for (int j = 0; j < F; ++j) {
        const double omega = w[j];
        std::vector<std::complex<double>> x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = std::complex<double>(ts[i], 0.0) * std::exp(std::complex<double>(0.0, omega * (double)i));
        }
        T[j] = cs_s1(x);
    }
    return T;
}

// ---- nearest index in sorted w for target ω (0-based) ----
static int nearest_index_in_w(const std::vector<double>& w, double target) {
    if (w.empty()) return -1;
    auto it = std::lower_bound(w.begin(), w.end(), target);
    if (it == w.begin()) return 0;
    if (it == w.end()) return (int)w.size() - 1;
    int hi = (int)std::distance(w.begin(), it);
    int lo = hi - 1;
    return (std::fabs(w[lo] - target) <= std::fabs(w[hi] - target)) ? lo : hi;
}

// ---- coarse grid: √n target frequencies mapped to dense grid (return 1-based) ----
static std::vector<int> build_coarse_indices(int n, const std::vector<double>& w_dense) {
    const int F0 = std::max(1, (int)std::floor(std::sqrt((double)n)));
    const double dwc = M_PI / (double)F0; // π/√n
    std::vector<int> idx;
    idx.reserve(F0);
    int last = -1;
    for (int k = 1; k <= F0; ++k) {
        double omega = k * dwc;
        int j0 = nearest_index_in_w(w_dense, omega); // 0-based
        if (j0 < 0) continue;
        if (j0 == last) continue;
        last = j0;
        idx.push_back(j0 + 1); // 1-based
    }
    return idx;
}

// ---- quantile with sorting + linear interpolation (R-like) ----
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

// ---- moving-sum helper for complex data (length m windows) ----
static void moving_sum_complex(const std::vector<std::complex<double>>& x, int m, std::vector<std::complex<double>>& out) {
    const int n = (int)x.size();
    const int rows = n - m + 1;
    out.assign(rows, std::complex<double>(0.0, 0.0));
    if (rows <= 0) return;
    std::complex<double> s(0.0, 0.0);
    for (int i = 0; i < m; ++i) s += x[i];
    out[0] = s;
    for (int i = m; i < n; ++i) {
        s += x[i];
        s -= x[i - m];
        out[i - m + 1] = s;
    }
}

// ---- bootstrap threshold on a sub-grid w_sub (build y then wild bootstrap) ----
// Fix: in the same bootstrap iteration, generate one G[r] and reuse it for all frequencies j; take the maximum across frequencies.
static double compute_crit_on_grid(const std::vector<double>& ts, const std::vector<double>& w_sub, int n, int m, int K, double alpha) {
    const int Fsub = (int)w_sub.size();
    if (Fsub == 0) return 0.0;
    const int rows = n - m + 1;
    // 1) Precompute Y[j][r] = window sums (corresponding to E_{r,m}(ω_j) in the paper)
    std::vector<std::vector<std::complex<double>>> Y(Fsub);
#pragma omp parallel for schedule(static)
    for (int j = 0; j < Fsub; ++j) {
        const double wj = w_sub[j];
        std::vector<std::complex<double>> x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = std::complex<double>(ts[i], 0.0) * std::exp(std::complex<double>(0.0, wj * (double)i));
        }
        moving_sum_complex(x, m, Y[j]);
    }

    // 2) Multiplier bootstrap (OBMB): within each bootstrap iteration, share one G across frequencies and take the maximum
    std::vector<double> row_max(K, 0.0);
#pragma omp parallel {
    std::random_device rd;
    unsigned seed = rd();
#ifdef _OPENMP
    seed ^= (unsigned)omp_get_thread_num() * 0x9e3779b9U;
#endif
    std::mt19937 gen(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<double> G(rows);
    std::vector<std::complex<double>> tmp(rows);
#pragma omp for schedule(static)
    for (int k = 0; k < K; ++k) {
        // 2.1 Generate one G and reuse across all frequencies
        for (int r = 0; r < rows; ++r) G[r] = N01(gen);
        double mx = -std::numeric_limits<double>::infinity();
        // 2.2 For each frequency: tmp[r] = Y[j][r] * G[r]; then compute prefix maximum
        for (int j = 0; j < Fsub; ++j) {
            for (int r = 0; r < rows; ++r) {
                tmp[r] = Y[j][r] * G[r];
            }
            double stat = cs_s1(tmp);
            if (stat > mx) mx = stat;
        }
        row_max[k] = mx;
    }
}
    std::sort(row_max.begin(), row_max.end());
    const double q = quantile_interp_sorted(row_max, 1.0 - alpha);
    return q / std::sqrt((double)m * (double)(n - m));
}

// ---- build refined indices union for ALL coarse-hot centers (±√n window, ≤1000/center) ----
// Fix: window radius s = floor(sqrt(n))
[[maybe_unused]] static std::vector<int> build_refined_union_multi(const std::vector<int>& coarse_hot_1b, int n, int F, int per_center_max = 1000) {
    if (F <= 0 || coarse_hot_1b.empty()) return {};
    const int s = std::max(1, (int)std::floor(std::sqrt((double)n)));
    std::vector<int> all;
    for (int c1 : coarse_hot_1b) {
        int L = std::max(1, c1 - s);
        int R = std::min(F, c1 + s);
        int count = R - L + 1;
        if (count <= 0) continue;
        if (count <= per_center_max) {
            for (int j = L; j <= R; ++j) all.push_back(j);
        } else {
            double step = (double)count / (double)per_center_max;
            int last = -1;
            for (int m = 0; m < per_center_max; ++m) {
                int j = L + (int)std::floor(m * step);
                if (j > R) j = R;
                if (j != last) {
                    all.push_back(j);
                    last = j;
                }
            }
            if (all.back() != R) all.push_back(R);
        }
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all; // 1-based indices in dense grid
}

// ---- neighborhood removal by frequency distance (1-based in/out) ----
[[maybe_unused]] static std::vector<int> prune_neighborhood(const std::vector<int>& keep_1b, const std::vector<int>& selected_1b, const std::vector<double>& w, double d_rad) {
    std::vector<int> out;
    out.reserve(keep_1b.size());
    for (int j1 : keep_1b) {
        double wi = w[j1 - 1];
        bool bad = false;
        for (int s1 : selected_1b) {
            if (std::fabs(wi - w[s1 - 1]) <= d_rad) {
                bad = true;
                break;
            }
        }
        if (!bad) out.push_back(j1);
    }
    return out;
}

