#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "../../LASS/src/LASS.h"
__global__ void LPCombFilterGPU(float *inputSample, float* outputSample, float inputGain, long inputDelay, float inputLpf_gain, float *delaybuf0, float *delaybuf1, long sampleSize);
__global__ void HexAllPassFilterGPU(float *inputSample, float *inputSample0, float *inputSample1, float *inputSample2, float *inputSample3, float *inputSample4, float *inputSample5, float* outputSample, float* envData, float inputGain, long inputDelay, float *delaybuf0, float *delaybuf1, long sampleSize);
__global__ void getEnvData(float *xyPoints, int *segmentTypes, float *envData, int segmentSize, long sampleSize);
SoundSample* do_reverb_SoundSample_GPU(SoundSample *inWave, Envelope *percentReverbinput, LPCombFilter **lpCombFilter, AllPassFilter *allPassFilter);
struct __align__(16) AR2Node {
    float a00, a01, a10, a11;
    float b0, b1;
};
SoundSample* do_biquad_filter_GPU(
    SoundSample *inWave, 
    float ba0, float ba1, float ba2, 
    float ba3, float ba4);
__device__ __forceinline__
AR2Node compose_nodes(const AR2Node& left, const AR2Node& right);

 
__global__ void AddPrefixAndExtract(
    AR2Node* __restrict__ nodes,
    const AR2Node* __restrict__ block_prefixes,
    float* __restrict__ output,
    long N);
__global__ void BiQuadFilterFused_Scan(
    const float* __restrict__ inputSample,
    AR2Node* __restrict__ outputNodes,
    AR2Node* __restrict__ block_results,
    float ba0, float ba1, float ba2,
    float alpha1, float alpha2,
    long sampleSize);
    __global__ void block_scan_kogge_stone(
        AR2Node* __restrict__ data,
        AR2Node* __restrict__ block_results,
        long N);