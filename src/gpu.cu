#include "vdb/gpu.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cfloat>
#include <numeric>
#include <random>
#include <vector>

// ============================================================
// Anonymous namespace: CUDA kernels
// ============================================================
namespace {

constexpr int TILE = 32;

// ----- Shared kernels (brute-force + IVF) -----

// Pairwise L2 squared-distance: each thread computes one element of (rows × cols).
__global__ void compute_l2_distances(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ dists,
                                     std::size_t rows, std::size_t cols,
                                     std::size_t dim) {
    std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;

    const float* a = A + r * dim;
    const float* b = B + c * dim;
    float sum = 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        float diff = a[j] - b[j];
        sum += diff * diff;
    }
    dists[r * cols + c] = sum;
}

// Max-heap push (device).  Used by both brute-force and IVF top-k.
__device__ void heap_push(float* heap_dist, uint32_t* heap_idx,
                          int k, int& heap_size,
                          float dist, uint32_t idx) {
    if (heap_size < k) {
        int pos = heap_size++;
        heap_dist[pos] = dist;
        heap_idx[pos] = idx;
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (heap_dist[pos] > heap_dist[parent]) {
                float td = heap_dist[pos]; heap_dist[pos] = heap_dist[parent]; heap_dist[parent] = td;
                uint32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[parent]; heap_idx[parent] = ti;
                pos = parent;
            } else break;
        }
    } else if (dist < heap_dist[0]) {
        heap_dist[0] = dist;
        heap_idx[0] = idx;
        int pos = 0;
        for (;;) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int largest = pos;
            if (left < k && heap_dist[left] > heap_dist[largest]) largest = left;
            if (right < k && heap_dist[right] > heap_dist[largest]) largest = right;
            if (largest == pos) break;
            float td = heap_dist[pos]; heap_dist[pos] = heap_dist[largest]; heap_dist[largest] = td;
            uint32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[largest]; heap_idx[largest] = ti;
            pos = largest;
        }
    }
}

// Extract sorted ascending from a max-heap into output arrays.
__device__ void heap_extract_sorted(float* heap_dist, uint32_t* heap_idx,
                                    int& heap_size, int k,
                                    float* o_dist, uint32_t* o_idx) {
    int out_count = heap_size < k ? heap_size : k;
    for (int i = out_count - 1; i >= 0; --i) {
        o_idx[i] = heap_idx[0];
        o_dist[i] = heap_dist[0];
        heap_dist[0] = heap_dist[heap_size - 1];
        heap_idx[0] = heap_idx[heap_size - 1];
        heap_size--;
        int pos = 0;
        for (;;) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int largest = pos;
            if (left < heap_size && heap_dist[left] > heap_dist[largest]) largest = left;
            if (right < heap_size && heap_dist[right] > heap_dist[largest]) largest = right;
            if (largest == pos) break;
            float td = heap_dist[pos]; heap_dist[pos] = heap_dist[largest]; heap_dist[largest] = td;
            uint32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[largest]; heap_idx[largest] = ti;
            pos = largest;
        }
    }
    for (int i = out_count; i < k; ++i) {
        o_idx[i] = 0;
        o_dist[i] = FLT_MAX;
    }
}

// ----- Brute-force top-k kernel -----

__global__ void find_topk(const float* __restrict__ dists,
                          uint32_t* __restrict__ out_indices,
                          float* __restrict__ out_distances,
                          std::size_t n, std::size_t k) {
    std::size_t qi = blockIdx.x;
    const float* row = dists + qi * n;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int ik = static_cast<int>(k > 256 ? 256 : k);

    float local_dist[256];
    uint32_t local_idx[256];
    int local_size = 0;

    for (std::size_t i = tid; i < n; i += nthreads) {
        heap_push(local_dist, local_idx, ik, local_size, row[i], static_cast<uint32_t>(i));
    }

    extern __shared__ char smem[];
    float* s_dist = reinterpret_cast<float*>(smem);
    uint32_t* s_idx = reinterpret_cast<uint32_t*>(s_dist + nthreads * ik);
    int* s_sizes = reinterpret_cast<int*>(s_idx + nthreads * ik);

    for (int i = 0; i < local_size; ++i) {
        s_dist[tid * ik + i] = local_dist[i];
        s_idx[tid * ik + i] = local_idx[i];
    }
    s_sizes[tid] = local_size;
    __syncthreads();

    if (tid == 0) {
        float merge_dist[256];
        uint32_t merge_idx[256];
        int merge_size = 0;

        for (int t = 0; t < nthreads; ++t) {
            int sz = s_sizes[t];
            for (int i = 0; i < sz; ++i) {
                heap_push(merge_dist, merge_idx, ik, merge_size,
                          s_dist[t * ik + i], s_idx[t * ik + i]);
            }
        }

        heap_extract_sorted(merge_dist, merge_idx, merge_size, static_cast<int>(k),
                            out_distances + qi * k, out_indices + qi * k);
    }
}

// ----- GPU k-means kernels -----

// Assign each vector to its nearest centroid.
// One thread per vector.  Writes assignment[i] = argmin centroid.
__global__ void kmeans_assign(const float* __restrict__ data,
                              const float* __restrict__ centroids,
                              uint32_t* __restrict__ assignments,
                              std::size_t n, std::size_t nlist,
                              std::size_t dim) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* vec = data + i * dim;
    float best_dist = FLT_MAX;
    uint32_t best_c = 0;

    for (std::size_t c = 0; c < nlist; ++c) {
        const float* cent = centroids + c * dim;
        float sum = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float diff = vec[d] - cent[d];
            sum += diff * diff;
        }
        if (sum < best_dist) {
            best_dist = sum;
            best_c = static_cast<uint32_t>(c);
        }
    }
    assignments[i] = best_c;
}

// Accumulate vector components into per-centroid sums and counts.
// Uses atomicAdd for both float sums and int counts.
__global__ void kmeans_accumulate(const float* __restrict__ data,
                                  const uint32_t* __restrict__ assignments,
                                  float* __restrict__ sums,
                                  uint32_t* __restrict__ counts,
                                  std::size_t n, std::size_t dim) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t c = assignments[i];
    atomicAdd(&counts[c], 1u);
    const float* vec = data + i * dim;
    float* sum = sums + c * dim;
    for (std::size_t d = 0; d < dim; ++d) {
        atomicAdd(&sum[d], vec[d]);
    }
}

// Update centroids: divide sums by counts.
// One thread per centroid dimension.
__global__ void kmeans_update(float* __restrict__ centroids,
                              const float* __restrict__ sums,
                              const uint32_t* __restrict__ counts,
                              const float* __restrict__ init_centroids,
                              std::size_t nlist, std::size_t dim) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t c = idx / dim;
    std::size_t d = idx % dim;
    if (c >= nlist) return;

    if (counts[c] == 0) {
        centroids[c * dim + d] = init_centroids[c * dim + d];
    } else {
        centroids[c * dim + d] = sums[c * dim + d] / static_cast<float>(counts[c]);
    }
}

// ----- GPU IVF search kernel -----

// One block per query.  Threads cooperatively scan vectors in probed lists,
// maintaining a shared top-k heap.
// probe_ids: [nq × nprobe] — centroid IDs to probe per query
// list_ids:  flattened vector IDs (sorted by centroid)
// list_offsets: [nlist+1] — start/end of each centroid's vectors in list_ids
__global__ void ivf_probe_search(const float* __restrict__ data,
                                 const float* __restrict__ queries,
                                 const uint32_t* __restrict__ probe_ids,
                                 const uint32_t* __restrict__ list_ids,
                                 const uint32_t* __restrict__ list_offsets,
                                 uint32_t* __restrict__ out_indices,
                                 float* __restrict__ out_distances,
                                 std::size_t dim, std::size_t k,
                                 std::size_t nprobe) {
    std::size_t qi = blockIdx.x;
    const float* q = queries + qi * dim;
    const uint32_t* my_probes = probe_ids + qi * nprobe;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int ik = static_cast<int>(k > 256 ? 256 : k);

    // Count total candidates across probed lists
    std::size_t total = 0;
    for (std::size_t p = 0; p < nprobe; ++p) {
        uint32_t c = my_probes[p];
        total += list_offsets[c + 1] - list_offsets[c];
    }

    // Thread-local heap
    float local_dist[256];
    uint32_t local_idx[256];
    int local_size = 0;

    // Linearize iteration across all probed lists
    for (std::size_t p = 0; p < nprobe; ++p) {
        uint32_t c = my_probes[p];
        uint32_t start = list_offsets[c];
        uint32_t end = list_offsets[c + 1];
        for (uint32_t j = start + tid; j < end; j += nthreads) {
            uint32_t vid = list_ids[j];
            const float* vec = data + vid * dim;
            float sum = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = q[d] - vec[d];
                sum += diff * diff;
            }
            heap_push(local_dist, local_idx, ik, local_size, sum, vid);
        }
    }

    // Merge via shared memory (same pattern as brute-force top-k)
    extern __shared__ char smem[];
    float* s_dist = reinterpret_cast<float*>(smem);
    uint32_t* s_idx = reinterpret_cast<uint32_t*>(s_dist + nthreads * ik);
    int* s_sizes = reinterpret_cast<int*>(s_idx + nthreads * ik);

    for (int i = 0; i < local_size; ++i) {
        s_dist[tid * ik + i] = local_dist[i];
        s_idx[tid * ik + i] = local_idx[i];
    }
    s_sizes[tid] = local_size;
    __syncthreads();

    if (tid == 0) {
        float merge_dist[256];
        uint32_t merge_idx[256];
        int merge_size = 0;

        for (int t = 0; t < nthreads; ++t) {
            int sz = s_sizes[t];
            for (int i = 0; i < sz; ++i) {
                heap_push(merge_dist, merge_idx, ik, merge_size,
                          s_dist[t * ik + i], s_idx[t * ik + i]);
            }
        }

        heap_extract_sorted(merge_dist, merge_idx, merge_size, static_cast<int>(k),
                            out_distances + qi * k, out_indices + qi * k);
    }
}

// Per-query: find top-nprobe centroids from centroid distance row.
// One block per query, blockDim.x threads.
__global__ void select_top_probes(const float* __restrict__ centroid_dists,
                                  uint32_t* __restrict__ probe_ids,
                                  std::size_t nlist, std::size_t nprobe) {
    std::size_t qi = blockIdx.x;
    const float* row = centroid_dists + qi * nlist;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int ik = static_cast<int>(nprobe > 256 ? 256 : nprobe);

    float local_dist[256];
    uint32_t local_idx[256];
    int local_size = 0;

    for (std::size_t c = tid; c < nlist; c += nthreads) {
        heap_push(local_dist, local_idx, ik, local_size, row[c], static_cast<uint32_t>(c));
    }

    extern __shared__ char smem[];
    float* s_dist = reinterpret_cast<float*>(smem);
    uint32_t* s_idx = reinterpret_cast<uint32_t*>(s_dist + nthreads * ik);
    int* s_sizes = reinterpret_cast<int*>(s_idx + nthreads * ik);

    for (int i = 0; i < local_size; ++i) {
        s_dist[tid * ik + i] = local_dist[i];
        s_idx[tid * ik + i] = local_idx[i];
    }
    s_sizes[tid] = local_size;
    __syncthreads();

    if (tid == 0) {
        float merge_dist[256];
        uint32_t merge_idx[256];
        int merge_size = 0;

        for (int t = 0; t < nthreads; ++t) {
            int sz = s_sizes[t];
            for (int i = 0; i < sz; ++i) {
                heap_push(merge_dist, merge_idx, ik, merge_size,
                          s_dist[t * ik + i], s_idx[t * ik + i]);
            }
        }

        // Write probe IDs sorted ascending by distance
        uint32_t* out = probe_ids + qi * nprobe;
        int out_count = merge_size < static_cast<int>(nprobe) ? merge_size : static_cast<int>(nprobe);

        // Extract from heap in reverse (ascending distance order)
        for (int i = out_count - 1; i >= 0; --i) {
            out[i] = merge_idx[0];
            merge_dist[0] = merge_dist[merge_size - 1];
            merge_idx[0] = merge_idx[merge_size - 1];
            merge_size--;
            int pos = 0;
            for (;;) {
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                int largest = pos;
                if (left < merge_size && merge_dist[left] > merge_dist[largest]) largest = left;
                if (right < merge_size && merge_dist[right] > merge_dist[largest]) largest = right;
                if (largest == pos) break;
                float td = merge_dist[pos]; merge_dist[pos] = merge_dist[largest]; merge_dist[largest] = td;
                uint32_t ti = merge_idx[pos]; merge_idx[pos] = merge_idx[largest]; merge_idx[largest] = ti;
                pos = largest;
            }
        }
        for (int i = out_count; i < static_cast<int>(nprobe); ++i) {
            out[i] = 0;
        }
    }
}

} // anonymous namespace

// ============================================================
// Public API: vdb namespace
// ============================================================
namespace vdb {

// ----- GpuIvfContext definition (opaque to callers) -----

struct GpuIvfContext {
    float* d_data = nullptr;
    float* d_centroids = nullptr;
    uint32_t* d_list_ids = nullptr;
    uint32_t* d_list_offsets = nullptr;
    std::size_t n = 0;
    std::size_t dim = 0;
    std::size_t nlist = 0;
};

// ----- Brute-force k-NN (unchanged) -----

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

void gpu_knn_l2(const float* data, std::size_t n, std::size_t dim,
                const float* queries, std::size_t nq, std::size_t k,
                IndexId* out_indices, float* out_distances) {
    if (n == 0 || nq == 0 || k == 0) return;
    if (k > 256) k = 256;

    float* d_data = nullptr;
    float* d_queries = nullptr;
    float* d_dists = nullptr;
    uint32_t* d_out_idx = nullptr;
    float* d_out_dist = nullptr;

    cudaMalloc(&d_data, n * dim * sizeof(float));
    cudaMalloc(&d_queries, nq * dim * sizeof(float));
    cudaMalloc(&d_dists, nq * n * sizeof(float));
    cudaMalloc(&d_out_idx, nq * k * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, nq * k * sizeof(float));

    cudaMemcpy(d_data, data, n * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, queries, nq * dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((static_cast<unsigned>(n) + TILE - 1) / TILE,
              (static_cast<unsigned>(nq) + TILE - 1) / TILE);
    compute_l2_distances<<<grid, block>>>(d_queries, d_data, d_dists, nq, n, dim);

    int topk_threads = 256;
    int ik = static_cast<int>(k);
    std::size_t smem_bytes = topk_threads * ik * sizeof(float)
                           + topk_threads * ik * sizeof(uint32_t)
                           + topk_threads * sizeof(int);
    find_topk<<<static_cast<unsigned>(nq), topk_threads, smem_bytes>>>(
        d_dists, d_out_idx, d_out_dist, n, k);

    cudaMemcpy(out_indices, d_out_idx, nq * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_distances, d_out_dist, nq * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_queries);
    cudaFree(d_dists);
    cudaFree(d_out_idx);
    cudaFree(d_out_dist);
}

// ----- GPU IVF: build -----

GpuIvfContext* gpu_ivf_build(const float* data, std::size_t n, std::size_t dim,
                             std::size_t nlist, std::size_t kmeans_iters,
                             float* out_centroids,
                             IndexId* out_assignments) {
    if (n == 0 || dim == 0 || nlist == 0) return nullptr;

    auto* ctx = new GpuIvfContext();
    ctx->n = n;
    ctx->dim = dim;
    ctx->nlist = nlist;

    // Allocate device memory for data (persistent)
    cudaMalloc(&ctx->d_data, n * dim * sizeof(float));
    cudaMemcpy(ctx->d_data, data, n * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device centroids
    cudaMalloc(&ctx->d_centroids, nlist * dim * sizeof(float));

    // Initialize centroids by random selection (same seed as CPU for reproducibility)
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<float> h_centroids(nlist * dim);
    for (std::size_t c = 0; c < nlist; ++c) {
        const float* src = data + indices[c] * dim;
        std::copy(src, src + dim, h_centroids.data() + c * dim);
    }
    cudaMemcpy(ctx->d_centroids, h_centroids.data(), nlist * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Keep a copy of initial centroids for empty-cluster reinit
    float* d_init_centroids = nullptr;
    cudaMalloc(&d_init_centroids, nlist * dim * sizeof(float));
    cudaMemcpy(d_init_centroids, ctx->d_centroids, nlist * dim * sizeof(float), cudaMemcpyDeviceToDevice);

    // Temp buffers for k-means
    uint32_t* d_assignments = nullptr;
    float* d_sums = nullptr;
    uint32_t* d_counts = nullptr;
    cudaMalloc(&d_assignments, n * sizeof(uint32_t));
    cudaMalloc(&d_sums, nlist * dim * sizeof(float));
    cudaMalloc(&d_counts, nlist * sizeof(uint32_t));

    int assign_threads = 256;
    int assign_blocks = static_cast<int>((n + assign_threads - 1) / assign_threads);
    int update_total = static_cast<int>(nlist * dim);
    int update_blocks = (update_total + assign_threads - 1) / assign_threads;

    for (std::size_t iter = 0; iter < kmeans_iters; ++iter) {
        // Assign vectors to nearest centroid
        kmeans_assign<<<assign_blocks, assign_threads>>>(
            ctx->d_data, ctx->d_centroids, d_assignments, n, nlist, dim);

        // Clear sums and counts
        cudaMemset(d_sums, 0, nlist * dim * sizeof(float));
        cudaMemset(d_counts, 0, nlist * sizeof(uint32_t));

        // Accumulate
        kmeans_accumulate<<<assign_blocks, assign_threads>>>(
            ctx->d_data, d_assignments, d_sums, d_counts, n, dim);

        // Update centroids
        kmeans_update<<<update_blocks, assign_threads>>>(
            ctx->d_centroids, d_sums, d_counts, d_init_centroids, nlist, dim);
    }

    // Final assignment to build inverted lists
    kmeans_assign<<<assign_blocks, assign_threads>>>(
        ctx->d_data, ctx->d_centroids, d_assignments, n, nlist, dim);

    // Copy assignments and centroids back to host
    std::vector<uint32_t> h_assignments(n);
    cudaMemcpy(h_assignments.data(), d_assignments, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids.data(), ctx->d_centroids, nlist * dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::copy(h_centroids.begin(), h_centroids.end(), out_centroids);
    for (std::size_t i = 0; i < n; ++i) {
        out_assignments[i] = h_assignments[i];
    }

    // Build flattened inverted lists on host, then upload
    // Count per list
    std::vector<uint32_t> counts(nlist, 0);
    for (std::size_t i = 0; i < n; ++i) counts[h_assignments[i]]++;

    // Offsets (prefix sum)
    std::vector<uint32_t> offsets(nlist + 1, 0);
    for (std::size_t c = 0; c < nlist; ++c) offsets[c + 1] = offsets[c] + counts[c];

    // Fill list IDs sorted by centroid
    std::vector<uint32_t> list_ids(n);
    std::vector<uint32_t> pos(nlist, 0);
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t c = h_assignments[i];
        list_ids[offsets[c] + pos[c]] = static_cast<uint32_t>(i);
        pos[c]++;
    }

    // Upload inverted lists to GPU (persistent)
    cudaMalloc(&ctx->d_list_ids, n * sizeof(uint32_t));
    cudaMalloc(&ctx->d_list_offsets, (nlist + 1) * sizeof(uint32_t));
    cudaMemcpy(ctx->d_list_ids, list_ids.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_list_offsets, offsets.data(), (nlist + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Free temp buffers
    cudaFree(d_assignments);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_init_centroids);

    return ctx;
}

// ----- GPU IVF: search -----

void gpu_ivf_search(const GpuIvfContext* ctx,
                    const float* queries, std::size_t nq,
                    std::size_t k, std::size_t nprobe,
                    IndexId* out_indices, float* out_distances) {
    if (!ctx || nq == 0 || k == 0) return;
    if (k > 256) k = 256;
    if (nprobe > ctx->nlist) nprobe = ctx->nlist;

    // Upload queries
    float* d_queries = nullptr;
    cudaMalloc(&d_queries, nq * ctx->dim * sizeof(float));
    cudaMemcpy(d_queries, queries, nq * ctx->dim * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Compute query-to-centroid distances (nq × nlist)
    float* d_centroid_dists = nullptr;
    cudaMalloc(&d_centroid_dists, nq * ctx->nlist * sizeof(float));

    dim3 block(TILE, TILE);
    dim3 grid((static_cast<unsigned>(ctx->nlist) + TILE - 1) / TILE,
              (static_cast<unsigned>(nq) + TILE - 1) / TILE);
    compute_l2_distances<<<grid, block>>>(
        d_queries, ctx->d_centroids, d_centroid_dists, nq, ctx->nlist, ctx->dim);

    // Step 2: Select top-nprobe centroids per query
    uint32_t* d_probe_ids = nullptr;
    cudaMalloc(&d_probe_ids, nq * nprobe * sizeof(uint32_t));

    int probe_threads = 128;
    int ip = static_cast<int>(nprobe > 256 ? 256 : nprobe);
    std::size_t probe_smem = probe_threads * ip * sizeof(float)
                           + probe_threads * ip * sizeof(uint32_t)
                           + probe_threads * sizeof(int);
    select_top_probes<<<static_cast<unsigned>(nq), probe_threads, probe_smem>>>(
        d_centroid_dists, d_probe_ids, ctx->nlist, nprobe);

    // Step 3: IVF probe search — one block per query
    uint32_t* d_out_idx = nullptr;
    float* d_out_dist = nullptr;
    cudaMalloc(&d_out_idx, nq * k * sizeof(uint32_t));
    cudaMalloc(&d_out_dist, nq * k * sizeof(float));

    int search_threads = 256;
    int ik = static_cast<int>(k > 256 ? 256 : k);
    std::size_t search_smem = search_threads * ik * sizeof(float)
                            + search_threads * ik * sizeof(uint32_t)
                            + search_threads * sizeof(int);
    ivf_probe_search<<<static_cast<unsigned>(nq), search_threads, search_smem>>>(
        ctx->d_data, d_queries, d_probe_ids, ctx->d_list_ids, ctx->d_list_offsets,
        d_out_idx, d_out_dist, ctx->dim, k, nprobe);

    // Copy results back
    cudaMemcpy(out_indices, d_out_idx, nq * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_distances, d_out_dist, nq * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free per-search temp buffers (data stays on GPU)
    cudaFree(d_queries);
    cudaFree(d_centroid_dists);
    cudaFree(d_probe_ids);
    cudaFree(d_out_idx);
    cudaFree(d_out_dist);
}

// ----- GPU IVF: free -----

void gpu_ivf_free(GpuIvfContext* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_data);
    cudaFree(ctx->d_centroids);
    cudaFree(ctx->d_list_ids);
    cudaFree(ctx->d_list_offsets);
    delete ctx;
}

} // namespace vdb
