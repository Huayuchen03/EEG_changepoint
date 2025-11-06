#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ===== External function declaration (defined elsewhere) =====
// This function performs the main frequency estimation algorithm.
std::vector<double> get_est_cpp(const std::vector<double>& time_series,
                                int n_w,
                                double alpha,
                                int K);

// ===== Function: Read the first column of CSV in streaming mode =====
// Reads the first column of data with overlap handling.
// Stores results in `segment`, uses `carry` to overlap between chunks.
// Returns false when the end of file is reached.
static bool read_next_segment_firstcol(
    std::ifstream& in,
    std::vector<double>& carry,
    size_t chunk_len,
    size_t overlap,
    std::vector<double>& segment,
    bool& header_checked,
    std::ofstream& log
) {
    segment.clear();
    segment.reserve(carry.size() + chunk_len);
    if (!carry.empty()) segment.insert(segment.end(), carry.begin(), carry.end());

    std::string line;
    line.reserve(256);

    while (segment.size() < chunk_len && std::getline(in, line)) {
        if (line.empty()) continue;

        size_t pos = line.find(',');
        const char* begin = line.c_str();
        const char* endptr = nullptr;
        double val = 0.0;

        if (pos != std::string::npos) {
            char saved = line[pos];
            const_cast<char&>(line[pos]) = '\0';
            val = std::strtod(begin, const_cast<char**>(&endptr));
            const_cast<char&>(line[pos]) = saved;
        } else {
            val = std::strtod(begin, const_cast<char**>(&endptr));
        }

        if (endptr == begin) {
            if (!header_checked) { header_checked = true; continue; }
            else continue;
        }

        header_checked = true;
        segment.push_back(val);
    }

    // Stop reading when end of file is reached and no new data is added.
    if (in.eof() && segment.size() <= carry.size()) {
        log << "[read] EOF reached, segment.size=" << segment.size()
            << " carry.size=" << carry.size() << " -> stop\n";
        log.flush();
        carry.clear();
        return false;
    }
    if (segment.empty()) {
        log << "[read] segment empty, stop.\n";
        log.flush();
        carry.clear();
        return false;
    }

    // Keep part of the end for the next overlap.
    const size_t keep = std::min(overlap, segment.size());
    carry.assign(segment.end() - keep, segment.end());
    return true;
}

// ===== Function: Merge close frequencies =====
// Combines frequencies that are too close to each other into one.
static std::vector<double> merge_close_freqs(std::vector<double> ws, int n_w) {
    if (ws.empty()) return {};
    std::sort(ws.begin(), ws.end());
    ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
    const double step = M_PI / (double)(n_w + 1);
    const double tol  = 2.5 * step;

    std::vector<double> out;
    double cur = ws[0];
    for (size_t i = 1; i < ws.size(); ++i) {
        if (std::fabs(ws[i] - cur) <= tol)
            cur = 0.5 * (cur + ws[i]);
        else {
            out.push_back(cur);
            cur = ws[i];
        }
    }
    out.push_back(cur);
    return out;
}

// ===== Main Program =====
// Reads CSV, processes data in chunks, runs get_est_cpp, and logs progress.
int main(int argc, char* argv[]) {
    std::ofstream log("progress.log", std::ios::app);
    if (!log) {
        std::cerr << "Cannot open log file progress.log\n";
        return 1;
    }

    if (argc < 2) {
        log << "Usage: " << argv[0]
            << " data.csv [chunk_len=5000] [overlap=300] [K=1000] [n_w=1000] [out_txt=result.txt]\n";
        return 1;
    }

    const std::string data_fn = argv[1];
    size_t chunk_len = 5000;
    size_t overlap   = 300;
    int    K         = 1000;
    int    n_w       = 1000;
    std::string out_txt = "result.txt";
    const double alpha = 0.05;

    if (argc >= 3) try { chunk_len = std::stoull(argv[2]); } catch (...) {}
    if (argc >= 4) try { overlap   = std::stoull(argv[3]); } catch (...) {}
    if (argc >= 5) try { K         = std::stoi(argv[4]);   } catch (...) {}
    if (argc >= 6) try { n_w       = std::stoi(argv[5]);   } catch (...) {}
    if (argc >= 7) out_txt = argv[6];

    if (chunk_len == 0 || overlap >= chunk_len || K <= 0 || n_w <= 0) {
        log << "Invalid arguments.\n";
        return 1;
    }

    std::ifstream dataf(data_fn);
    if (!dataf) {
        log << "Cannot open data file: " << data_fn << "\n";
        return 1;
    }

    log << "== Processing started ==\n"
        << "File: " << data_fn << "\n"
        << "chunk_len=" << chunk_len << ", overlap=" << overlap
        << ", K=" << K << ", n_w=" << n_w << "\n";

    std::vector<double> carry, segment, all_omegas;
    bool header_checked = false;
    size_t chunk_idx = 0;

    // Read and process data segment by segment.
    while (read_next_segment_firstcol(dataf, carry, chunk_len, overlap, segment, header_checked, log)) {
        ++chunk_idx;
        const size_t n = segment.size();

        // Log the last two values in this segment.
        if (!segment.empty()) {
            size_t sz = segment.size();
            double last1 = segment[sz - 1];
            double last2 = (sz > 1 ? segment[sz - 2] : last1);
            log << "[chunk " << chunk_idx << "] last values: "
                << std::setprecision(10) << last2 << ", " << last1 << "\n";
            log.flush();
        }

        if (n < 3) {
            log << "[chunk " << chunk_idx << "] skipped (too short)\n";
            log.flush();
            continue;
        }

#ifdef _OPENMP
        log << "[chunk " << chunk_idx << "] threads=" << omp_get_max_threads() << "\n";
#endif
        log.flush();

        // Run the external frequency estimation function.
        std::vector<double> omegas = get_est_cpp(segment, n_w, alpha, K);
        log << "[chunk " << chunk_idx << "] Done, detected=" << omegas.size() << "\n";
        log.flush();

        // Store detected frequencies for later merging.
        if (!omegas.empty())
            all_omegas.insert(all_omegas.end(), omegas.begin(), omegas.end());

        // Stop if segment is too small or EOF is reached.
        if (segment.size() <= overlap) {
            log << "[chunk " << chunk_idx << "] segment <= overlap, break.\n";
            break;
        }
        if (dataf.eof()) {
            log << "[chunk " << chunk_idx << "] EOF reached, break.\n";
            break;
        }
    }

    // Merge all detected frequencies and write to result file.
    std::vector<double> final_omegas = merge_close_freqs(std::move(all_omegas), n_w);
    std::ofstream out(out_txt, std::ios::trunc);
    if (out) {
        out.precision(16);
        for (double w : final_omegas) out << w << "\n";
        out.close();
    }

    log << "== Finished == wrote " << final_omegas.size()
        << " frequency(ies) to " << out_txt << "\n";
    log.close();
    return 0;
}

