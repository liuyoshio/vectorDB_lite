#include "vdb/hnsw.hpp"
#include "vdb/io.hpp"
#include "vdb/ivf.hpp"
#include "vdb/types.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage() {
  std::cout
      << "vdb_cli search --index ivf|hnsw --data data.bin --queries queries.bin --k K [options]\n"
      << "\n"
      << "Options:\n"
      << "  --out PATH              Write results as CSV (qi,rank,id,dist)\n"
      << "  --use-gpu               Enable GPU brute-force search (if built with CUDA)\n"
      << "  --nlist N               IVF number of lists (default 1024)\n"
      << "  --kmeans-iters N         IVF kmeans iterations (default 20)\n"
      << "  --nprobe N              IVF probes (default 8)\n"
      << "  --M N                   HNSW max neighbors (default 16)\n"
      << "  --ef-construction N     HNSW construction ef (default 200)\n"
      << "  --ef-search N           HNSW search ef (default 64)\n"
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

  if (command != "search") {
    std::cerr << "Unknown command: " << command << "\n";
    print_usage();
    return 1;
  }

  const char* index_name = get_arg_value(args, "--index");
  const char* data_path = get_arg_value(args, "--data");
  const char* query_path = get_arg_value(args, "--queries");
  const char* k_value = get_arg_value(args, "--k");
  const char* out_path = get_arg_value(args, "--out");

  if (!index_name || !data_path || !query_path || !k_value) {
    std::cerr << "Missing required arguments.\n";
    print_usage();
    return 1;
  }

  std::size_t k = to_size(k_value, 0);
  if (k == 0) {
    std::cerr << "k must be > 0\n";
    return 1;
  }

  try {
    vdb::Dataset data = vdb::load_f32_matrix(data_path);
    vdb::Dataset queries = vdb::load_f32_matrix(query_path);

    if (data.dim() != queries.dim()) {
      std::cerr << "data dim != query dim\n";
      return 1;
    }

    vdb::SearchParams params;
    params.use_gpu = has_flag(args, "--use-gpu");

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
      for (std::size_t qi = 0; qi < result.nq; ++qi) {
        std::cout << "query " << qi << ":";
        for (std::size_t j = 0; j < result.k; ++j) {
          std::size_t idx = qi * result.k + j;
          std::cout << " " << result.indices[idx] << "(" << result.distances[idx] << ")";
        }
        std::cout << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
