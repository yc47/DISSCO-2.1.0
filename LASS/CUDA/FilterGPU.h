#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "../../LASS/src/LASS.h"
__global__ void LPCombFilterGPU(float *inputSample, float* outputSample, float inputGain, long inputDelay, float inputLpf_gain, float *delaybuf0, float *delaybuf1, long sampleSize);
__global__ void HexAllPassFilterGPU(float *inputSample, float *inputSample0, float *inputSample1, float *inputSample2, float *inputSample3, float *inputSample4, float *inputSample5, float* outputSample, float* envData, float inputGain, long inputDelay, float *delaybuf0, float *delaybuf1, long sampleSize);
__global__ void getEnvData(float *xyPoints, int *segmentTypes, float *envData, int segmentSize, long sampleSize);
SoundSample* do_reverb_SoundSample_GPU(SoundSample *inWave, Envelope *percentReverbinput, LPCombFilter **lpCombFilter, AllPassFilter *allPassFilter);
class AR2Scan {
public:
    using value_type = float;

    
    struct __align__(32) Node {
        value_type a00, a01, a10, a11; // A
        value_type b0,  b1;            // b
    };

    struct compose_nodes {
        __host__ __device__
        Node operator()(const Node& left, const Node& right) const {
            Node out;
            // A = Ar * Al
            out.a00 = right.a00*left.a00 + right.a01*left.a10;
            out.a01 = right.a00*left.a01 + right.a01*left.a11;
            out.a10 = right.a10*left.a00 + right.a11*left.a10;
            out.a11 = right.a10*left.a01 + right.a11*left.a11;
            // b = Ar*bl + br
            out.b0  = right.a00*left.b0 + right.a01*left.b1 + right.b0;
            out.b1  = right.a10*left.b0 + right.a11*left.b1 + right.b1;
            return out;
        }
    };

    struct make_node_from_y {
        value_type a, b;
        __host__ __device__
        Node operator()(const value_type yi) const {
            Node n;
            n.a00 = a;   n.a01 = b;
            n.a10 = 1;   n.a11 = 0;
            n.b0  = yi;  n.b1  = 0;
            return n;
        }
    };

    struct get_b0 {
        __host__ __device__
        value_type operator()(const Node& n) const { return n.b0; }
    };
    static value_type getb0(Node node){
        return node.b0;
    }
    
};
__global__ void BiQuadFilterFused(
    const float* __restrict__ inputSample,
    AR2Scan::Node* __restrict__ outputNodes,
    float ba0, float ba1, float ba2,
    float alpha1, float alpha2,
    long sampleSize);
SoundSample* do_biquad_filter_GPU(
    SoundSample *inWave, 
    float ba0, float ba1, float ba2, 
    float ba3, float ba4);