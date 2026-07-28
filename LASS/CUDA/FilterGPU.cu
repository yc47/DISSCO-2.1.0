
#include "FilterGPU.h"
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#define CEIL_MULT(x, y)  ( (( (x) + (y) - 1 ) / (y) ) * (y) )

// Optimized node structure with 16-byte alignment
 

// Device function for composing two AR2 nodes
__device__ __forceinline__
AR2Node compose_nodes(const AR2Node& left, const AR2Node& right) {
    AR2Node out;
    // Matrix multiplication: out.A = right.A * left.A
    out.a00 = right.a00 * left.a00 + right.a01 * left.a10;
    out.a01 = right.a00 * left.a01 + right.a01 * left.a11;
    out.a10 = right.a10 * left.a00 + right.a11 * left.a10;
    out.a11 = right.a10 * left.a01 + right.a11 * left.a11;
    // Vector: out.b = right.A * left.b + right.b
    out.b0 = right.a00 * left.b0 + right.a01 * left.b1 + right.b0;
    out.b1 = right.a10 * left.b0 + right.a11 * left.b1 + right.b1;
    return out;
}

// KERNEL 1: Merged feedforward computation, node creation, and block scan
__global__ void BiQuadFilterFused_Scan(
    const float* __restrict__ inputSample,
    AR2Node* __restrict__ outputNodes,
    AR2Node* __restrict__ block_results,
    float ba0, float ba1, float ba2,
    float alpha1, float alpha2,
    long sampleSize)
{
    // Double-buffered so each scan step needs one __syncthreads() instead of
    // two: writes always land in the buffer nobody is reading this step, so
    // there's no read-after-write hazard to guard against.
    __shared__ AR2Node temp[2][256];

    int tid = threadIdx.x;
    long idx = (long)blockIdx.x * blockDim.x + tid;

    AR2Node initial;
    // Compute feedforward and create node
    if (idx < sampleSize) {
        float x_n  = inputSample[idx];
        float x_n1 = (idx >= 1) ? inputSample[idx - 1] : 0.0f;
        float x_n2 = (idx >= 2) ? inputSample[idx - 2] : 0.0f;

        float f_n = ba0 * x_n + ba1 * x_n1 + ba2 * x_n2;

        initial.a00 = alpha1; initial.a01 = alpha2;
        initial.a10 = 1.0f;   initial.a11 = 0.0f;
        initial.b0 = f_n;     initial.b1 = 0.0f;
    } else {
        // Out of bounds - identity node
        initial.a00 = 1.0f; initial.a01 = 0.0f;
        initial.a10 = 0.0f; initial.a11 = 1.0f;
        initial.b0 = 0.0f;  initial.b1 = 0.0f;
    }

    int pout = 0;
    temp[pout][tid] = initial;
    __syncthreads();

    // Kogge-Stone inclusive scan within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int pin = pout;
        pout = 1 - pout;
        if (tid >= stride) {
            temp[pout][tid] = compose_nodes(temp[pin][tid - stride], temp[pin][tid]);
        } else {
            temp[pout][tid] = temp[pin][tid];
        }
        __syncthreads();
    }

    AR2Node result = temp[pout][tid];

    // Write back to global memory
    if (idx < sampleSize) {
        outputNodes[idx] = result;
    }

    // Save last element of each block for inter-block scan
    if (tid == blockDim.x - 1 && block_results != nullptr) {
        block_results[blockIdx.x] = result;
    }
}

// KERNEL 2: Simple Kogge-Stone inclusive scan (for block results)
__global__ void block_scan_kogge_stone(
    AR2Node* __restrict__ data,
    AR2Node* __restrict__ block_results,
    long N)
{
    __shared__ AR2Node temp[2][256];

    int tid = threadIdx.x;
    long idx = (long)blockIdx.x * blockDim.x + tid;

    AR2Node initial;
    // Load into shared memory
    if (idx < N) {
        initial = data[idx];
    } else {
        // Out of bounds - shouldn't affect results
        initial.a00 = 1.0f; initial.a01 = 0.0f;
        initial.a10 = 0.0f; initial.a11 = 1.0f;
        initial.b0 = 0.0f;  initial.b1 = 0.0f;
    }

    int pout = 0;
    temp[pout][tid] = initial;
    __syncthreads();

    // Kogge-Stone inclusive scan (double-buffered: one sync per step)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int pin = pout;
        pout = 1 - pout;
        if (tid >= stride) {
            temp[pout][tid] = compose_nodes(temp[pin][tid - stride], temp[pin][tid]);
        } else {
            temp[pout][tid] = temp[pin][tid];
        }
        __syncthreads();
    }

    AR2Node result = temp[pout][tid];

    // Write back to global memory
    if (idx < N) {
        data[idx] = result;
    }

    // Save last element of each block for inter-block scan
    if (tid == blockDim.x - 1 && block_results != nullptr) {
        block_results[blockIdx.x] = result;
    }
}

// KERNEL 3: Add block prefix and extract output (b0 component)
__global__ void AddPrefixAndExtract(
    AR2Node* __restrict__ nodes,
    const AR2Node* __restrict__ block_prefixes,
    float* __restrict__ output,
    long N)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Add block prefix if not in first block
        AR2Node node = nodes[idx];
        if (blockIdx.x > 0) {
            node = compose_nodes(block_prefixes[blockIdx.x - 1], node);
        }

        // Extract b0 to output
        output[idx] = node.b0;
    }
}

// KERNEL 3b: Same prefix composition as AddPrefixAndExtract, but keeps the
// full AR2Node instead of extracting b0. Used to fold a higher scan level's
// results back into a lower level's block sums, which is what lets the
// multi-level scan in scanNodesRecursive() handle arbitrarily many blocks.
__global__ void AddBlockPrefixInPlace(
    AR2Node* __restrict__ nodes,
    const AR2Node* __restrict__ block_prefixes,
    long N)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N && blockIdx.x > 0) {
        nodes[idx] = compose_nodes(block_prefixes[blockIdx.x - 1], nodes[idx]);
    }
}

// Total number of extra AR2Node slots scanNodesIterative() needs as scratch
// space to scan an array of `numBlocks` elements: one block-sums array per
// scan level, for every level beyond the first that doesn't fit in a single
// block. Mirrors the level sizes scanNodesIterative() will actually walk
// through, so the caller can size a single arena big enough up front.
static long computeScanScratchNodes(long numBlocks, int threadsPerBlock)
{
    long extra = 0;
    long count = numBlocks;
    while (count > threadsPerBlock) {
        long blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        extra += blocks;
        count = blocks;
    }
    return extra;
}

// Scans an AR2Node array of arbitrary length in place, turning it into a
// fully-correct inclusive scan regardless of how many blocks it takes.
//
// block_scan_kogge_stone() only scans correctly *within* a block: with N
// elements it produces `blocks = ceil(N/threadsPerBlock)` independent scan
// runs whose boundaries haven't been reconciled with each other yet. This
// walks down level by level (each level's block sums becoming the next
// level's input) until everything fits in one block, then walks back up
// folding each level's now-correct prefixes into the level below it - the
// standard technique for scanning arrays larger than one block (as in
// NVIDIA's "Parallel Prefix Sum" algorithm / Thrust's multi-level scan).
//
// d_scratch must point to at least computeScanScratchNodes(N, threadsPerBlock)
// AR2Nodes; the caller carves it out of a persistent arena so this never
// calls cudaMalloc/cudaFree itself. Recursion depth is O(log_256(N)), so 8
// levels covers any realistic (or unrealistic) sample count.
static void scanNodesIterative(AR2Node* d_data, long N, int threadsPerBlock, AR2Node* d_scratch)
{
    if (N <= 1) return; // a single element is already its own inclusive scan

    AR2Node* levelData[8];
    long levelN[8];
    int depth = 0;

    AR2Node* curData = d_data;
    long curN = N;
    AR2Node* scratchCursor = d_scratch;

    // Down-sweep: scan each level's blocks, carving each level's block-sum
    // array out of the pre-sized scratch arena, until one level's worth of
    // block sums fits in a single block.
    while (curN > threadsPerBlock) {
        long blocks = (curN + threadsPerBlock - 1) / threadsPerBlock;
        AR2Node* nextData = scratchCursor;
        scratchCursor += blocks;

        block_scan_kogge_stone<<<(int)blocks, threadsPerBlock>>>(curData, nextData, curN);

        levelData[depth] = curData;
        levelN[depth] = curN;
        depth++;

        curData = nextData;
        curN = blocks;
    }

    // Top level fits in one block: scanning it in place is globally correct.
    block_scan_kogge_stone<<<1, threadsPerBlock>>>(curData, nullptr, curN);

    // Up-sweep: fold each level's fully-correct prefixes back into the level
    // below it, from the top level back down to the original array.
    for (int i = depth - 1; i >= 0; i--) {
        long blocks = (levelN[i] + threadsPerBlock - 1) / threadsPerBlock;
        AddBlockPrefixInPlace<<<(int)blocks, threadsPerBlock>>>(levelData[i], curData, levelN[i]);
        curData = levelData[i];
        curN = levelN[i];
    }
}

// Persistent device scratch space for do_biquad_filter_GPU, grown on demand
// and reused across calls instead of cudaMalloc/cudaFree'd every time.
// Profiling (see Reverb::do_reverb_SoundSample's gpu/cpu timers) showed the
// ~13 CUDA driver calls a single invocation used to make - 3 cudaMallocs,
// 3 cudaFrees, 5 kernel launches, 2 memcpys - dominating wall-clock time for
// typical buffer sizes, since the actual scan kernels finish in low
// microseconds once the data fits in the GPU's L2 cache. Reusing these
// arenas cuts steady-state calls down to just the memcpys and kernel
// launches - no allocation calls at all once the arena has grown to the
// largest sampleSize seen so far.
//
// Not thread-safe: do_biquad_filter_GPU is only ever called from a single
// host thread in the current CMOD/LASSIE pipeline (reverb is applied
// sequentially per track). If that changes, this needs a per-thread arena
// or a mutex around the ensure*Arena() calls.
namespace {
    float* g_floatArena = nullptr;
    long g_floatArenaCapacity = 0; // in floats

    AR2Node* g_nodeArena = nullptr;
    long g_nodeArenaCapacity = 0; // in AR2Node elements

    float* ensureFloatArena(long neededFloats)
    {
        if (neededFloats > g_floatArenaCapacity) {
            if (g_floatArena != nullptr) {
                CUDA_CHECK(cudaFree(g_floatArena));
            }
            CUDA_CHECK(cudaMalloc(&g_floatArena, neededFloats * sizeof(float)));
            g_floatArenaCapacity = neededFloats;
        }
        return g_floatArena;
    }

    AR2Node* ensureNodeArena(long neededNodes)
    {
        if (neededNodes > g_nodeArenaCapacity) {
            if (g_nodeArena != nullptr) {
                CUDA_CHECK(cudaFree(g_nodeArena));
            }
            CUDA_CHECK(cudaMalloc(&g_nodeArena, neededNodes * sizeof(AR2Node)));
            g_nodeArenaCapacity = neededNodes;
        }
        return g_nodeArena;
    }
}

// Host function: Complete biquad filter with custom kernels only
SoundSample* do_biquad_filter_GPU(
    SoundSample *inWave,
    float ba0, float ba1, float ba2,
    float ba3, float ba4)
{
    long sampleSize = inWave->getSampleCount();
    SoundSample *outWave = new SoundSample(sampleSize, inWave->getSamplingRate());

    // Calculate grid dimensions
    const int threadsPerBlock = 256;
    const int numBlocks = (int)((sampleSize + threadsPerBlock - 1) / threadsPerBlock);
    const long scanScratchNodes = computeScanScratchNodes(numBlocks, threadsPerBlock);

    // Grab (and grow, if needed) the persistent arenas instead of allocating
    // fresh device memory every call.
    float *d_input, *d_output;
    float* d_float_base = ensureFloatArena(2 * sampleSize);
    d_input = d_float_base;                    // First sampleSize floats
    d_output = d_float_base + sampleSize;      // Next sampleSize floats

    AR2Node *d_nodes, *d_block_results, *d_scan_scratch;
    AR2Node* d_node_base = ensureNodeArena(sampleSize + numBlocks + scanScratchNodes);
    d_nodes = d_node_base;                          // First sampleSize nodes
    d_block_results = d_node_base + sampleSize;     // Next numBlocks nodes
    d_scan_scratch = d_block_results + numBlocks;   // Remaining scratch for scanNodesIterative

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, inWave->getData(),
                          sampleSize * sizeof(float), cudaMemcpyHostToDevice));

    // STEP 1: Merged feedforward + node creation + per-block scan.
    // No cudaDeviceSynchronize() here: kernels launched into the same
    // (default) stream already execute in issued order on the GPU, so the
    // host doesn't need to block and wait between them - only before reading
    // the result back, which the final cudaMemcpy below already does.
    BiQuadFilterFused_Scan<<<numBlocks, threadsPerBlock>>>(
        d_input, d_nodes, d_block_results, ba0, ba1, ba2, -ba3, -ba4, sampleSize
    );

    // STEP 2: Fully scan the block results, however many levels that takes.
    scanNodesIterative(d_block_results, numBlocks, threadsPerBlock, d_scan_scratch);

    // STEP 3: Add the now-correct block prefixes and extract output
    AddPrefixAndExtract<<<numBlocks, threadsPerBlock>>>(
        d_nodes, d_block_results, d_output, sampleSize
    );

    // Copy result back to host (blocking - this is what drains the stream)
    CUDA_CHECK(cudaMemcpy(outWave->getData(), d_output,
                          sampleSize * sizeof(float), cudaMemcpyDeviceToHost));

    return outWave;
}
// Reproduces LPCombFilter::do_filter's coupled recurrence in parallel.
// CPU reference (LPCombFilter.cpp + LowPassFilter.cpp):
//   y[n] = x[n-D] + g*L[n-D]           (n >= D, else y[n] = 0)
//   L[n] = y[n] + lpf_gain*L[n-1]      (L[-1] = 0)
// where D is the comb delay and L is the internal lowpass filter's state,
// advanced by exactly one step per comb call.
//
// Substituting y[n] shows L only depends on values one delay-block back
// (L[n-D], via y[n-D]) and one step back (L[n-1]), so it can be computed one
// delay-length block at a time: given L for block (j-1), block j's y values
// are a plain elementwise formula, and L for block j is a single-pole scan
// of those y values seeded with L's last value from block (j-1). That scan
// is what the Hillis-Steele doubling loop below computes.
//
// The previous version of this kernel got the block decomposition right but
// carried the wrong quantity between blocks (the comb's *output* value
// instead of the lowpass filter's *internal state* L, and added it after
// computing y instead of folding it into the scan's seed beforehand) -
// confirmed wrong by the correctness check in Reverb::do_reverb_SoundSample.
// LPCombFilterGPU used to do everything (the setup, every block's elementwise
// step, and every block's Hillis-Steele scan) inside one <<<1,256>>> launch,
// which caps the kernel at 256 GPU threads total regardless of delay length
// or sample count - for a delay in the thousands and a sample count in the
// hundreds of thousands, that meant one thread doing over a thousand
// sequential loop iterations while the other ~16000 cores on the GPU sat
// idle. Splitting each step into its own kernel, launched with
// gridDim = ceil(delay/256), lets every block-local step use as many blocks
// as the delay actually needs. The block-to-block sequential dependency
// (block j needs block j-1's fully-scanned result) is preserved by simply
// launching each block's kernels after the previous block's, in the same
// stream - CUDA guarantees same-stream kernels run in launch order, which is
// actually simpler to reason about than the old single-kernel version's
// __syncthreads()-based bookkeeping.

// Zero-fills output block 0 and sets Zsrc/output block 1 from the input,
// exactly as LPComb's first two blocks are always 0 and a direct passthrough
// of the input (see do_lp_filter_GPU's comment for the math).
__global__ void LPCombSetup(
    const float* __restrict__ inputSample,
    float* __restrict__ outputSample,
    float* __restrict__ Zsrc,
    long delay,
    long sampleSize)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < delay) {
        if (idx < sampleSize) {
            outputSample[idx] = 0.0f;
        }
        float v = (idx < sampleSize) ? inputSample[idx] : 0.0f;
        Zsrc[idx] = v;
        if (idx + delay < sampleSize) {
            outputSample[idx + delay] = v;
        }
    }
}

// Folds the previous block's L-carry into this block's scan seed, before the
// scan below runs. Single-thread: only Zsrc[0] is touched.
__global__ void LPCombFoldCarry(float* __restrict__ Zsrc, const float* __restrict__ carry, float lpf_gain)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Zsrc[0] += lpf_gain * (*carry);
    }
}

// One doubling round of the Hillis-Steele scan for the block's L recursion.
__global__ void LPCombScanRound(
    const float* __restrict__ src,
    float* __restrict__ dst,
    float gaine,
    long off,
    long delay)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < delay) {
        dst[idx] = (idx >= off) ? (src[idx] + gaine * src[idx - off]) : src[idx];
    }
}

// Saves this block's last scanned L value as the carry for the next block.
// Launched before LPCombConvertToY below in the same stream, so this always
// reads the scan result before ConvertToY starts overwriting it - no
// explicit sync needed, same-stream launch order already guarantees it.
__global__ void LPCombSaveCarry(const float* __restrict__ Zsrc, float* __restrict__ carry, long delay)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *carry = Zsrc[delay - 1];
    }
}

// Converts this block's scanned L values into the next block's y (comb
// output) values, and writes them to the output buffer.
__global__ void LPCombConvertToY(
    float* __restrict__ Zsrc,
    const float* __restrict__ inputSample,
    float* __restrict__ outputSample,
    float gain,
    long j,
    long delay,
    long sampleSize)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < delay) {
        float v = gain * Zsrc[idx] + inputSample[(j - 1) * delay + idx];
        Zsrc[idx] = v;
        if (j * delay + idx < sampleSize) {
            outputSample[j * delay + idx] = v;
        }
    }
}

// Runs the full LPComb filter on device buffers the caller already owns
// (do_lp_filter_GPU and do_reverb_SoundSample_GPU each have their own buffer
// lifetime/reuse strategy, so this just takes pointers rather than owning
// any memory itself). Zsrc/Zdest must each be at least
// ceil(delay/256)*256 floats; carryScratch is a single device float used as
// scratch space across iterations, reset to 0 here at the start of every
// call (it does not need to persist between calls).
//
// Math (from LPCombFilter.cpp + LowPassFilter.cpp):
//   y[n] = x[n-D] + g*L[n-D]        (n >= D, else y[n] = 0)
//   L[n] = y[n] + lpf_gain*L[n-1]   (L[-1] = 0)
// Substituting y[n] shows L only depends on values one delay-block back
// (via y[n-D]) and one step back (L[n-1]), so it's computed one delay-length
// block at a time: given L for block (j-1), block j's y values are a plain
// elementwise formula, and L for block j is a scan of those y values seeded
// with L's last value from block (j-1).
static void runLPCombFilterGPU(
    const float* d_input,
    float* d_output,
    float* d_Zsrc,
    float* d_Zdest,
    float* d_carryScratch,
    float gain,
    long delay,
    float lpf_gain,
    long sampleSize)
{
    const int threadsPerBlock = 256;
    const int delayBlocks = (int)((delay + threadsPerBlock - 1) / threadsPerBlock);
    const long delayPadded = (long)delayBlocks * threadsPerBlock;

    CUDA_CHECK(cudaMemsetAsync(d_carryScratch, 0, sizeof(float)));

    LPCombSetup<<<delayBlocks, threadsPerBlock>>>(d_input, d_output, d_Zsrc, delay, sampleSize);

    long ps = (sampleSize + delay - 1) / delay;
    float* Zsrc = d_Zsrc;
    float* Zdest = d_Zdest;

    for (long j = 2; j < ps; ++j) {
        LPCombFoldCarry<<<1, 1>>>(Zsrc, d_carryScratch, lpf_gain);

        float gaine = lpf_gain;
        for (long off = 1; off < delayPadded; off *= 2) {
            LPCombScanRound<<<delayBlocks, threadsPerBlock>>>(Zsrc, Zdest, gaine, off, delay);
            float* temp = Zsrc; Zsrc = Zdest; Zdest = temp;
            gaine *= gaine;
        }
        // Zsrc now holds L for block j-1.

        LPCombSaveCarry<<<1, 1>>>(Zsrc, d_carryScratch, delay);
        LPCombConvertToY<<<delayBlocks, threadsPerBlock>>>(Zsrc, d_input, d_output, gain, j, delay, sampleSize);
    }
}

namespace {
    float* g_lpInputArena = nullptr;
    float* g_lpOutputArena = nullptr;
    long g_lpFloatArenaCapacity = 0; // shared capacity for input/output (both sampleSize)

    float* g_lpZsrcArena = nullptr;
    float* g_lpZdestArena = nullptr;
    long g_lpDelayArenaCapacity = 0; // shared capacity for Zsrc/Zdest (both delayPadded)

    float* g_lpCarryArena = nullptr; // single float, allocated once

    void ensureLPArenas(long sampleSize, long delayPadded, float** outInput, float** outOutput, float** outZsrc, float** outZdest, float** outCarry)
    {
        if (sampleSize > g_lpFloatArenaCapacity) {
            if (g_lpInputArena != nullptr) CUDA_CHECK(cudaFree(g_lpInputArena));
            if (g_lpOutputArena != nullptr) CUDA_CHECK(cudaFree(g_lpOutputArena));
            CUDA_CHECK(cudaMalloc(&g_lpInputArena, sampleSize * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_lpOutputArena, sampleSize * sizeof(float)));
            g_lpFloatArenaCapacity = sampleSize;
        }
        if (delayPadded > g_lpDelayArenaCapacity) {
            if (g_lpZsrcArena != nullptr) CUDA_CHECK(cudaFree(g_lpZsrcArena));
            if (g_lpZdestArena != nullptr) CUDA_CHECK(cudaFree(g_lpZdestArena));
            CUDA_CHECK(cudaMalloc(&g_lpZsrcArena, delayPadded * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_lpZdestArena, delayPadded * sizeof(float)));
            g_lpDelayArenaCapacity = delayPadded;
        }
        if (g_lpCarryArena == nullptr) {
            CUDA_CHECK(cudaMalloc(&g_lpCarryArena, sizeof(float)));
        }
        *outInput = g_lpInputArena;
        *outOutput = g_lpOutputArena;
        *outZsrc = g_lpZsrcArena;
        *outZdest = g_lpZdestArena;
        *outCarry = g_lpCarryArena;
    }
}

SoundSample* do_lp_filter_GPU(SoundSample *inWave, float lpf_g, float g, long d){
    long sampleSize = inWave->getSampleCount();
    SoundSample *outWave = new SoundSample(sampleSize, inWave->getSamplingRate());

    const int threadsPerBlock = 256;
    const long delayPadded = (long)((d + threadsPerBlock - 1) / threadsPerBlock) * threadsPerBlock;

    float *d_input, *d_output, *d_Zsrc, *d_Zdest, *d_carry;
    ensureLPArenas(sampleSize, delayPadded, &d_input, &d_output, &d_Zsrc, &d_Zdest, &d_carry);

    CUDA_CHECK(cudaMemcpy(d_input, inWave->getData(), sampleSize * sizeof(float), cudaMemcpyHostToDevice));

    runLPCombFilterGPU(d_input, d_output, d_Zsrc, d_Zdest, d_carry, g, d, lpf_g, sampleSize);

    CUDA_CHECK(cudaMemcpy(outWave->getData(), d_output, sampleSize * sizeof(float), cudaMemcpyDeviceToHost));
    return outWave;
}

__global__ void HexAllPassFilterGPU(float *inputSample, float *inputSample0, float *inputSample1, float *inputSample2, float *inputSample3, float *inputSample4, float *inputSample5, float* outputSample, float* envData, float inputGain, long inputDelay, float *delaybuf0, float *delaybuf1, long sampleSize){
    float gain=inputGain, gsqrd=gain*gain, x;
    int tx = blockIdx.x*blockDim.x+threadIdx.x, idx;
    long delay = inputDelay, ps=(double)(sampleSize+delay-1)/delay, pb=(double)(delay+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
    int stridesz=(delay+blockDim.x-1)/blockDim.x*blockDim.x;
    //__shared__ float X[4096], Y[4096];

    for (int i = 0; i < pb; ++i){
        idx = i * gridDim.x * blockDim.x + tx;
        if (idx < delay){
            delaybuf0[stridesz*blockIdx.x+blockDim.x*i+threadIdx.x] = (inputSample0[idx]+inputSample1[idx]+inputSample2[idx]+inputSample3[idx]+inputSample4[idx]+inputSample5[idx])/6;
            delaybuf1[stridesz*blockIdx.x+blockDim.x*i+threadIdx.x] = -gain*delaybuf0[stridesz*blockIdx.x+blockDim.x*i+threadIdx.x];
            outputSample[idx] = delaybuf1[stridesz*blockIdx.x+blockDim.x*i+threadIdx.x]*envData[idx] + (1-envData[idx])*inputSample[idx];
        }
    }

    for(int i=1; i<ps; ++i){
        for (int j = 0; j < pb; ++j){
            idx = j* gridDim.x * blockDim.x + tx;
            if (idx < delay&& i*delay+idx < sampleSize){
                x=delaybuf0[stridesz*blockIdx.x+blockDim.x*j+threadIdx.x];
                delaybuf0[stridesz*blockIdx.x+blockDim.x*j+threadIdx.x] = (inputSample0[i*delay+idx]+inputSample1[i*delay+idx]+inputSample2[i*delay+idx]+inputSample3[i*delay+idx]+inputSample4[i*delay+idx]+inputSample5[i*delay+idx])/6;
                delaybuf1[stridesz*blockIdx.x+blockDim.x*j+threadIdx.x] = -gain*delaybuf0[stridesz*blockIdx.x+blockDim.x*j+threadIdx.x]+(1-gsqrd)*(gain*delaybuf1[stridesz*blockIdx.x+blockDim.x*j+threadIdx.x]+x);
                outputSample[i*delay+idx] = delaybuf1[stridesz*blockIdx.x+blockDim.x*j+threadIdx.x]*envData[i*delay+idx]+(1-envData[i*delay+idx])*inputSample[i*delay+idx];
            }
        }
    }
}
// The old AllPassFilterGPU did its whole doubling scan (stride = D, 2D, 4D,
// ...) inside one <<<1,256>>> launch with an in-kernel __syncthreads()
// between rounds - which only synchronizes within a single block, so it
// silently capped the kernel at 256 threads total no matter how large
// sampleSize was (256 threads is a small fraction of an RTX-class GPU).
// Unlike LPCombFilterGPU, this scan runs across the *entire* sample buffer
// in one pass (no per-delay-block decomposition), so each round is already
// embarrassingly parallel across all of sampleSize - it just needs enough
// blocks to cover it. Moving each round to its own kernel launch (one round
// = one launch, host-driven loop below) lets it use
// gridDim = ceil(sampleSize/256) blocks per round instead of being capped at
// 256 threads, and turns the round count into O(log2(sampleSize/D)) kernel
// launches - typically single digits to a few dozen, not the thousand-plus
// sequential iterations LPCombFilterGPU's block-recurrent structure needs.

// Initial elementwise setup: b0[idx] = b1[idx] = -g*x[idx] + c1*x[idx-D] (for
// idx >= D). Both buffers start identical; the round loop below scans them.
__global__ void AllPassFilterInit(
    const float* __restrict__ inputSample,
    float* __restrict__ b0,
    float* __restrict__ b1,
    float g,
    float c1,
    long D,
    long sampleSize)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sampleSize) {
        float v = -g * inputSample[idx];
        if (idx >= D) {
            v += c1 * inputSample[idx - D];
        }
        b0[idx] = v;
        b1[idx] = v;
    }
}

// One doubling round of the D-strided Hillis-Steele-style scan.
__global__ void AllPassFilterRound(
    const float* __restrict__ b0,
    float* __restrict__ b1,
    float m,
    long stride,
    long sampleSize)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sampleSize) {
        b1[idx] = (idx >= stride) ? (b0[idx] + m * b0[idx - stride]) : b0[idx];
    }
}

namespace {
    float* g_apInputArena = nullptr;
    float* g_apB0Arena = nullptr;
    float* g_apB1Arena = nullptr;
    long g_apArenaCapacity = 0;

    void ensureAPArenas(long sampleSize, float** outInput, float** outB0, float** outB1)
    {
        if (sampleSize > g_apArenaCapacity) {
            if (g_apInputArena != nullptr) CUDA_CHECK(cudaFree(g_apInputArena));
            if (g_apB0Arena != nullptr) CUDA_CHECK(cudaFree(g_apB0Arena));
            if (g_apB1Arena != nullptr) CUDA_CHECK(cudaFree(g_apB1Arena));
            CUDA_CHECK(cudaMalloc(&g_apInputArena, sampleSize * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_apB0Arena, sampleSize * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g_apB1Arena, sampleSize * sizeof(float)));
            g_apArenaCapacity = sampleSize;
        }
        *outInput = g_apInputArena;
        *outB0 = g_apB0Arena;
        *outB1 = g_apB1Arena;
    }
}

SoundSample* do_ap_filter_GPU(SoundSample *inWave, float g, long d){

    long sampleSize = inWave->getSampleCount();
    float* inWaveData = inWave->getData();
    SoundSample *outWave = new SoundSample(sampleSize, inWave->getSamplingRate());

    const int threadsPerBlock = 256;
    const int blocks = (int)((sampleSize + threadsPerBlock - 1) / threadsPerBlock);

    float *d_input, *d_b0, *d_b1;
    ensureAPArenas(sampleSize, &d_input, &d_b0, &d_b1);

    CUDA_CHECK(cudaMemcpy(d_input, inWaveData, sampleSize * sizeof(float), cudaMemcpyHostToDevice));

    float c1 = 1 - g*g;
    AllPassFilterInit<<<blocks, threadsPerBlock>>>(d_input, d_b0, d_b1, g, c1, d, sampleSize);

    // Host tracks which buffer is current, so - unlike the old kernel's
    // ping-pong - there's no ambiguity about where the final result ends up
    // regardless of how many rounds run.
    float* cur = d_b0;
    float* next = d_b1;
    float m = c1 * g; // c2
    for (long stride = d; stride < (sampleSize+1)/2; stride *= 2) {
        AllPassFilterRound<<<blocks, threadsPerBlock>>>(cur, next, m, stride, sampleSize);
        float* temp = cur; cur = next; next = temp;
        m = m*m;
    }

    CUDA_CHECK(cudaMemcpy(outWave->getData(), cur,
                          sampleSize * sizeof(float), cudaMemcpyDeviceToHost));

    return outWave;
}


__global__ void getEnvData(float *xyPoints, int *segmentTypes, float *envData, int segmentSize, long sampleSize){
    int tx = blockDim.x*blockIdx.x+threadIdx.x, samples, idx, start, i, j;
    float x0, y0, x1, y1, m0=0, m1=0, t, a, b;
    for(i=0; i<segmentSize; ++i){
        x0=xyPoints[2*i];
        y0=xyPoints[2*i+1];
        x1=xyPoints[2*i+2];
        y1=xyPoints[2*i+3];
        samples=(x1-x0)*sampleSize;
        start=sampleSize * x0;
        if(segmentTypes[i]==2){

            m0=(y1-y0)/(x1-x0)/sampleSize;
            for(j=0; j<samples/(blockDim.x*gridDim.x); ++j){
                idx = j * (blockDim.x*gridDim.x)+ tx;
                envData[start + idx] = y0 + idx*m0;
            }

            idx = j * (blockDim.x*gridDim.x) + tx;
            if(start+ idx<sampleSize)
                envData[start + idx] = y0 + idx*m0;

        }
        else if(segmentTypes[i]==1){
            //Cubic Hermite spline
            if(i!=0){
                if(segmentTypes[i-1]==0){
                    a=y0>y1?3:-3;
                    b=pow(2.71828, (double)a);
                    m0=a*(y0-xyPoints[2*i-1])*b/(x0-xyPoints[2*i-2])/(b-1);
                }
                else if(segmentTypes[i-1]==1)
                    m0=(y1-xyPoints[2*i-1])/(x1-xyPoints[2*i-2]);
                else 
                    m0=(y0-xyPoints[2*i-1])/(x0-xyPoints[2*i-2]);
            }
            else
                m0=0;

            if(i+1<segmentSize){
                if(segmentTypes[i+1]==0){
                    a=xyPoints[2*i+5]>y1?3:-3;
                    m1=a*(xyPoints[2*i+5]-y1)/(xyPoints[2*i+4]-x1)/(pow(2.71828, (double)a)-1);
                }
                else if(segmentTypes[i+1]==1)
                    m1=(xyPoints[2*i+5]-y0)/(xyPoints[2*i+4]-x0);
                else 
                    m1=(xyPoints[2*i+5]-y1)/(xyPoints[2*i+4]-x1);
            }
            else
                m1=0;

            a= -3*y0+3*y1-2*m0-m1;
            b= 2*y0-2*y1+m0+m1;

            for(j=0; j<samples/(blockDim.x*gridDim.x); ++j){
                idx = j * (blockDim.x*gridDim.x) + tx;
                t=(float)idx/samples;
                envData[start + idx] = y0 + m0*t + a*t*t + b*t*t*t;
            }
            idx = j * (blockDim.x*gridDim.x) + tx;
            if(start + idx<sampleSize){
                t=(float)idx/samples;
                envData[start + idx] = y0 + m0*t + a*t*t + b*t*t*t;
            }
        }
        else{
            a=y1>y0?3:-3;
            b=pow(2.71828, (double)a);

            for(j=0; j<samples/(blockDim.x*gridDim.x); ++j){
                idx = j * (blockDim.x*gridDim.x) + tx;
                envData[start + idx] = y0+(y1-y0)*(1-pow(2.71828, (double)a*idx/samples))/(1-b);
            }

            idx = j * (blockDim.x*gridDim.x) + tx;
            if(start+ idx<sampleSize)
                envData[start + idx] = y0+(y1-y0)*(1-pow(2.71828, (double)a*idx/samples))/(1-b);
        }
        __syncthreads();
    }
}

void plotWithGnuplot(const std::vector<float>& data) {
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    if (gnuplotPipe) {
        // Set up the plot
        fprintf(gnuplotPipe, "set title 'Plot of Floats from 0 to 2'\n");
        fprintf(gnuplotPipe, "set xlabel 'Index'\n");
        fprintf(gnuplotPipe, "set ylabel 'Value'\n");
        fprintf(gnuplotPipe, "plot '-' with lines\n");
        
        // Send data to GNUplot
        for(size_t i = 0; i < data.size(); ++i){
            fprintf(gnuplotPipe, "%zu %f\n", i, data[i]);
        }
        fprintf(gnuplotPipe, "e\n");
        pclose(gnuplotPipe);
    } else {
        std::cerr << "Could not open pipe to GNUplot.\n";
    }
}

SoundSample* do_reverb_SoundSample_GPU(SoundSample *inWave, Envelope *percentReverbinput, LPCombFilter **lpCombFilter, AllPassFilter *allPassFilter){ 
    SoundSample *outWave=new SoundSample(inWave->getSampleCount(),inWave->getSamplingRate());
    float *inWaveData=inWave->getData(), *outWaveDataD0, *outWaveDataD1, *outWaveDataD2, *outWaveDataD3, *outWaveDataD4, *outWaveDataD5, *outWaveDataD, *inWaveDataD, *outWaveData=new float[inWave->getSampleCount()];
    float *delay0bufD0, *delay0bufD1, *delay0bufD2, *delay0bufD3, *delay0bufD4, *delay0bufD5;
    float *delay1bufD0, *delay1bufD1, *delay1bufD2, *delay1bufD3, *delay1bufD4, *delay1bufD5;
    float *delay0bufAllP, *delay1bufAllP;
    long sampleSize=inWave->getSampleCount();
    float durationofEnv=percentReverbinput->getDuration();
    float *envData=new float[sampleSize], *envDataD, *envXY, *envXYD;
    int *envSegType, *envSegTypeD, segSize;

    Collection<envelope_segment> *segs=percentReverbinput->getSegments();
    envelope_segment seg;
    segSize=segs->size();

    envXY=new float[segSize*2];
    envSegType=new int[segSize-1];

    for (int i = 1; i < segSize; ++i){
        seg=segs->get(i);
        envXY[i*2]=seg.x;
        envXY[i*2+1]=seg.y;
        envSegType[i-1]=seg.interType;
    }
    
    seg=segs->get(0);
    envXY[0]=seg.x;
    envXY[1]=seg.y;

    cudaMalloc(&inWaveDataD, sampleSize*sizeof(float));
    cudaMalloc(&outWaveDataD0, sampleSize*sizeof(float));
    cudaMalloc(&outWaveDataD1, sampleSize*sizeof(float));
    cudaMalloc(&outWaveDataD2, sampleSize*sizeof(float));
    cudaMalloc(&outWaveDataD3, sampleSize*sizeof(float));
    cudaMalloc(&outWaveDataD4, sampleSize*sizeof(float));
    cudaMalloc(&outWaveDataD5, sampleSize*sizeof(float));
    cudaMalloc(&delay0bufD0, CEIL_MULT(lpCombFilter[0]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay0bufD1, CEIL_MULT(lpCombFilter[1]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay0bufD2, CEIL_MULT(lpCombFilter[2]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay0bufD3, CEIL_MULT(lpCombFilter[3]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay0bufD4, CEIL_MULT(lpCombFilter[4]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay0bufD5, CEIL_MULT(lpCombFilter[5]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay1bufD0, CEIL_MULT(lpCombFilter[0]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay1bufD1, CEIL_MULT(lpCombFilter[1]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay1bufD2, CEIL_MULT(lpCombFilter[2]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay1bufD3, CEIL_MULT(lpCombFilter[3]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay1bufD4, CEIL_MULT(lpCombFilter[4]->get_D(), 256)*sizeof(float));
    cudaMalloc(&delay1bufD5, CEIL_MULT(lpCombFilter[5]->get_D(), 256)*sizeof(float));
    cudaMalloc(&outWaveDataD, sampleSize*sizeof(float));
    cudaMalloc(&envDataD, sampleSize*sizeof(float));
    cudaMemcpy(inWaveDataD, inWaveData, sampleSize*sizeof(float), cudaMemcpyHostToDevice);
    // Single small carry scratch buffer shared across all 6 comb filters -
    // safe because runLPCombFilterGPU resets it at the start of every call,
    // and these 6 calls are issued to the same (default) stream, so each
    // one's kernels fully complete before the next one's begin.
    float* lpCarryScratch;
    cudaMalloc(&lpCarryScratch, sizeof(float));
    runLPCombFilterGPU(inWaveDataD, outWaveDataD0, delay0bufD0, delay1bufD0, lpCarryScratch, lpCombFilter[0]->get_g(), lpCombFilter[0]->get_D(), lpCombFilter[0]->get_lpf_g(), sampleSize);
    runLPCombFilterGPU(inWaveDataD, outWaveDataD1, delay0bufD1, delay1bufD1, lpCarryScratch, lpCombFilter[1]->get_g(), lpCombFilter[1]->get_D(), lpCombFilter[1]->get_lpf_g(), sampleSize);
    runLPCombFilterGPU(inWaveDataD, outWaveDataD2, delay0bufD2, delay1bufD2, lpCarryScratch, lpCombFilter[2]->get_g(), lpCombFilter[2]->get_D(), lpCombFilter[2]->get_lpf_g(), sampleSize);
    runLPCombFilterGPU(inWaveDataD, outWaveDataD3, delay0bufD3, delay1bufD3, lpCarryScratch, lpCombFilter[3]->get_g(), lpCombFilter[3]->get_D(), lpCombFilter[3]->get_lpf_g(), sampleSize);
    runLPCombFilterGPU(inWaveDataD, outWaveDataD4, delay0bufD4, delay1bufD4, lpCarryScratch, lpCombFilter[4]->get_g(), lpCombFilter[4]->get_D(), lpCombFilter[4]->get_lpf_g(), sampleSize);
    runLPCombFilterGPU(inWaveDataD, outWaveDataD5, delay0bufD5, delay1bufD5, lpCarryScratch, lpCombFilter[5]->get_g(), lpCombFilter[5]->get_D(), lpCombFilter[5]->get_lpf_g(), sampleSize);
    cudaFree(lpCarryScratch);
    cudaDeviceSynchronize();

    cudaMalloc(&envXYD, segSize*2*sizeof(float));
    cudaMalloc(&envSegTypeD, (segSize-1)*sizeof(int));
    cudaMalloc(&delay0bufAllP, CEIL_MULT(allPassFilter->get_D(), 256)*6*sizeof(float));
    cudaMalloc(&delay1bufAllP, CEIL_MULT(allPassFilter->get_D(), 256)*6*sizeof(float));

    cudaMemcpy(envXYD, envXY, segSize*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(envSegTypeD, envSegType, (segSize-1)*sizeof(int), cudaMemcpyHostToDevice);

    getEnvData<<<6, 256>>>(envXYD, envSegTypeD, envDataD, segSize-1, sampleSize);
    cudaDeviceSynchronize();

    //cout<<"envXY "<<envXY[0]<<" "<<envXY[1]<<" "<<envXY[2]<<" "<<envXY[3]<<" "<<envXY[4]<<" "<<envXY[5]<<endl;
    //cout<<"envSegType "<<envSegType[0]<<" "<<envSegType[1]<<endl;
    //cout<<"segSize "<<segSize<<endl;
    
    //cudaMemcpy(envData, envDataD, sampleSize*sizeof(float), cudaMemcpyDeviceToHost);
    //std::vector<float> plot;
    //
    //for (int i = 0; i < sampleSize; i+=1000) {
    //    plot.push_back(envData[i]);
    //}
    //plotWithGnuplot(plot);

    HexAllPassFilterGPU<<<6, 256>>>(inWaveDataD, outWaveDataD0, outWaveDataD1, outWaveDataD2, outWaveDataD3, outWaveDataD4, outWaveDataD5, outWaveDataD, envDataD, allPassFilter->get_g(), allPassFilter->get_D(), delay0bufAllP, delay1bufAllP, sampleSize);

    cudaDeviceSynchronize();


    cudaMemcpy(outWave->getData(), outWaveDataD, sampleSize*sizeof(float), cudaMemcpyDeviceToHost);
    // cout<<"outwave 0 "<<(*outWave)[0]<<endl;
    // cout<<"outwave 1000 "<<(*outWave)[1000]<<endl;
    // cout<<"outwave 10000 "<<(*outWave)[10000]<<endl;
    // cout<<"outwave 100000 "<<(*outWave)[100000]<<endl;

    cudaFree(inWaveDataD);
    cudaFree(outWaveDataD0);
    cudaFree(outWaveDataD1);
    cudaFree(outWaveDataD2);
    cudaFree(outWaveDataD3);
    cudaFree(outWaveDataD4);
    cudaFree(outWaveDataD5);
    cudaFree(delay0bufD0);
    cudaFree(delay0bufD1);
    cudaFree(delay0bufD2);
    cudaFree(delay0bufD3);
    cudaFree(delay0bufD4);
    cudaFree(delay0bufD5);
    cudaFree(delay1bufD0);
    cudaFree(delay1bufD1);
    cudaFree(delay1bufD2);
    cudaFree(delay1bufD3);
    cudaFree(delay1bufD4);
    cudaFree(delay1bufD5);
    cudaFree(delay0bufAllP);
    cudaFree(delay1bufAllP);
    cudaFree(outWaveDataD);
    cudaFree(envDataD);
    cudaFree(envXYD);
    cudaFree(envSegTypeD);
    delete[] envData;
    delete[] outWaveData;
    delete[] envXY;
    delete[] envSegType;

    return outWave;
}
