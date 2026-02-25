#include "vdb/hnsw.hpp"
#include "vdb/io.hpp"
#include "vdb/ivf.hpp"
#include "vdb/types.hpp"
#include "vdb/gpu.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef VDB_USE_OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

void print_usage() {
  std::cout
      << "vdb_cli search --index ivf|hnsw --data data.bin --queries queries.bin --k K [options]\n"
      << "vdb_cli bench  --index ivf|hnsw --data data.bin --queries queries.bin --k K [options]\n"
      << "\n"
      << "Options:\n"
      << "  --out PATH              Write results as CSV (qi,rank,id,dist)\n"
      << "  --use-gpu               Force GPU search (CUDA builds only)\n"
      << "  --no-gpu                Force CPU search\n"
      << "  --auto-gpu              Auto-detect GPU (default when CUDA is enabled)\n"
      << "  --nlist N               IVF number of lists (default 1024)\n"
      << "  --kmeans-iters N        IVF kmeans iterations (default 20)\n"
      << "  --nprobe N              IVF probes (default 8)\n"
      << "  --M N                   HNSW max neighbors (default 16)\n"
      << "  --ef-construction N     HNSW construction ef (default 200)\n"
      << "  --ef-search N           HNSW search ef (default 64)\n"
      << "  --threads N             CPU threads for query parallelism (OpenMP builds)\n"
      << "  --repeat N              (bench) timed search repeats (default 3)\n"
      << "  --warmup N              (bench) warmup repeats (default 1)\n"
      << "\n"
      << "Binary format for data/queries:\n"
      << "  [uint32 count][uint32 dim][count*dim float32]\n";
}

const char* get_arg_value(const std::vector<std::string>& args, const std::string& key) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1].c_str();
    }
  }
  return nullptr;
}

bool has_flag(const std::vector<std::string>& args, const std::string& key) {
  for (const auto& a : args) {
    if (a == key) {
      return true;
    }
  }
  return false;
}

std::size_t to_size(const char* value, std::size_t fallback) {
  if (!value) {
    return fallback;
  }
  return static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
}

void print_results(const vdb::SearchResult& result) {
  for (std::size_t qi = 0; qi < result.nq; ++qi) {
    std::cout << "query " << qi << ":";
    for (std::size_t j = 0; j < result.k; ++j) {
      std::size_t idx = qi * result.k + j;
      std::cout << " " << result.indices[idx] << "(" << result.distances[idx] << ")";
    }
    std::cout << "\n";
  }
}

bool parse_common_required(const std::vector<std::string>& args, const char*& index_name,
                           const char*& data_path, const char*& query_path,
                           std::size_t& k, const char*& out_path) {
  index_name = get_arg_value(args, "--index");
  data_path = get_arg_value(args, "--data");
  query_path = get_arg_value(args, "--queries");
  const char* k_value = get_arg_value(args, "--k");
  out_path = get_arg_value(args, "--out");

  if (!index_name || !data_path || !query_path || !k_value) {
    std::cerr << "Missing required arguments.\n";
    print_usage();
    return false;
  }

  k = to_size(k_value, 0);
  if (k == 0) {
    std::cerr << "k must be > 0\n";
    return false;
  }

  return true;
}

bool resolve_use_gpu(const std::vector<std::string>& args) {
  bool force_on = has_flag(args, "--use-gpu");
  bool force_off = has_flag(args, "--no-gpu");
  bool auto_flag = has_flag(args, "--auto-gpu");

  if (force_on && force_off) {
    std::cerr << "warning: both --use-gpu and --no-gpu set; using CPU\n";
    return false;
  }

  if (force_off) {
    return false;
  }

#ifdef VDB_USE_CUDA
  bool available = vdb::gpu_available();
  if (force_on && !available) {
    std::cerr << "warning: GPU requested but no CUDA device found; falling back to CPU\n";
    return false;
  }
  if (force_on) {
    return true;
  }
  if (auto_flag || !force_on) {
    if (!available) {
      std::cerr << "info: no CUDA device detected; using CPU\n";
    }
    return available;
  }
  return false;
#else
  if (force_on || auto_flag) {
    std::cerr << "warning: GPU flags ignored because this build has no CUDA support\n";
  }
  return false;
#endif
}

int run_search_command(const std::vector<std::string>& args) {
  const char* index_name = nullptr;
  const char* data_path = nullptr;
  const char* query_path = nullptr;
  const char* out_path = nullptr;
  std::size_t k = 0;

  if (!parse_common_required(args, index_name, data_path, query_path, k, out_path)) {
    return 1;
  }

  vdb::Dataset data = vdb::load_f32_matrix(data_path);
  vdb::Dataset queries = vdb::load_f32_matrix(query_path);

  if (data.dim() != queries.dim()) {
    std::cerr << "data dim != query dim\n";
    return 1;
  }

  vdb::SearchParams params;
  params.use_gpu = resolve_use_gpu(args);

  std::string idx(index_name);
  vdb::SearchResult result;

  if (idx == "ivf") {
    vdb::IvfConfig cfg;
    cfg.nlist = to_size(get_arg_value(args, "--nlist"), cfg.nlist);
    cfg.kmeans_iters = to_size(get_arg_value(args, "--kmeans-iters"), cfg.kmeans_iters);

    vdb::IvfIndex index(cfg);
    index.build(data);

    params.nprobe = to_size(get_arg_value(args, "--nprobe"), params.nprobe);
    index.search(queries, k, result, params);
  } else if (idx == "hnsw") {
    vdb::HnswConfig cfg;
    cfg.M = to_size(get_arg_value(args, "--M"), cfg.M);
    cfg.ef_construction = to_size(get_arg_value(args, "--ef-construction"), cfg.ef_construction);
    cfg.ef_search = to_size(get_arg_value(args, "--ef-search"), cfg.ef_search);

    vdb::HnswIndex index(cfg);
    index.build(data);

    params.ef_search = to_size(get_arg_value(args, "--ef-search"), params.ef_search);
    index.search(queries, k, result, params);
  } else {
    std::cerr << "Unknown index: " << idx << "\n";
    return 1;
  }

  if (out_path) {
    vdb::save_search_results_text(out_path, result.indices, result.distances, result.nq, result.k);
  } else {
    print_results(result);
  }

  return 0;
}

int run_bench_command(const std::vector<std::string>& args) {
  const char* index_name = nullptr;
  const char* data_path = nullptr;
  const char* query_path = nullptr;
  const char* out_path = nullptr;
  std::size_t k = 0;

  if (!parse_common_required(args, index_name, data_path, query_path, k, out_path)) {
    return 1;
  }

  std::size_t repeat = to_size(get_arg_value(args, "--repeat"), 3);
  std::size_t warmup = to_size(get_arg_value(args, "--warmup"), 1);
  if (repeat == 0) {
    std::cerr << "repeat must be > 0\n";
    return 1;
  }

  vdb::Dataset data = vdb::load_f32_matrix(data_path);
  vdb::Dataset queries = vdb::load_f32_matrix(query_path);

  if (data.dim() != queries.dim()) {
    std::cerr << "data dim != query dim\n";
    return 1;
  }

  vdb::SearchParams params;
  params.use_gpu = resolve_use_gpu(args);

  std::string idx(index_name);
  vdb::SearchResult result;
  std::vector<double> search_secs;
  search_secs.reserve(repeat);

  auto t_build_start = Clock::now();
  if (idx == "ivf") {
    vdb::IvfConfig cfg;
    cfg.nlist = to_size(get_arg_value(args, "--nlist"), cfg.nlist);
    cfg.kmeans_iters = to_size(get_arg_value(args, "--kmeans-iters"), cfg.kmeans_iters);

    vdb::IvfIndex index(cfg);
    index.build(data);
    params.nprobe = to_size(get_arg_value(args, "--nprobe"), params.nprobe);

    auto t_build_end = Clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();

    for (std::size_t i = 0; i < warmup; ++i) {
      index.search(queries, k, result, params);
    }
    for (std::size_t i = 0; i < repeat; ++i) {
      auto t0 = Clock::now();
      index.search(queries, k, result, params);
      auto t1 = Clock::now();
      search_secs.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    double mean_search = std::accumulate(search_secs.begin(), search_secs.end(), 0.0) /
                         static_cast<double>(search_secs.size());
    double qps = static_cast<double>(queries.count()) / mean_search;

    std::cout << "METRIC build_sec " << build_sec << "\n";
    std::cout << "METRIC search_sec_mean " << mean_search << "\n";
    std::cout << "METRIC qps " << qps << "\n";
  } else if (idx == "hnsw") {
    vdb::HnswConfig cfg;
    cfg.M = to_size(get_arg_value(args, "--M"), cfg.M);
    cfg.ef_construction = to_size(get_arg_value(args, "--ef-construction"), cfg.ef_construction);
    cfg.ef_search = to_size(get_arg_value(args, "--ef-search"), cfg.ef_search);

    vdb::HnswIndex index(cfg);
    index.build(data);
    params.ef_search = to_size(get_arg_value(args, "--ef-search"), params.ef_search);

    auto t_build_end = Clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();

    for (std::size_t i = 0; i < warmup; ++i) {
      index.search(queries, k, result, params);
    }
    for (std::size_t i = 0; i < repeat; ++i) {
      auto t0 = Clock::now();
      index.search(queries, k, result, params);
      auto t1 = Clock::now();
      search_secs.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    double mean_search = std::accumulate(search_secs.begin(), search_secs.end(), 0.0) /
                         static_cast<double>(search_secs.size());
    double qps = static_cast<double>(queries.count()) / mean_search;

    std::cout << "METRIC build_sec " << build_sec << "\n";
    std::cout << "METRIC search_sec_mean " << mean_search << "\n";
    std::cout << "METRIC qps " << qps << "\n";
  } else {
    std::cerr << "Unknown index: " << idx << "\n";
    return 1;
  }

  if (out_path) {
    vdb::save_search_results_text(out_path, result.indices, result.distances, result.nq, result.k);
  }

  print_results(result);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }

  std::vector<std::string> args(argv + 1, argv + argc);
  std::string command = args[0];

  if (command == "help" || command == "--help" || command == "-h") {
    print_usage();
    return 0;
  }

  std::size_t threads = to_size(get_arg_value(args, "--threads"), 0);
  if (get_arg_value(args, "--threads") && threads == 0) {
    std::cerr << "threads must be > 0\n";
    return 1;
  }
#ifdef VDB_USE_OPENMP
  if (threads > 0) {
    omp_set_num_threads(static_cast<int>(threads));
  }
#else
  if (threads > 0) {
    std::cerr << "warning: --threads ignored because OpenMP is not enabled in this build\n";
  }
#endif

  try {
    if (command == "search") {
      return run_search_command(args);
    }
    if (command == "bench") {
      return run_bench_command(args);
    }

    std::cerr << "Unknown command: " << command << "\n";
    print_usage();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
