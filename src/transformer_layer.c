// Page 4: https://arxiv.org/abs/1506.04214v2
// Page 3: https://arxiv.org/pdf/1705.06368v3.pdf
// https://wikimedia.org/api/rest_v1/media/math/render/svg/1edbece2559479959fe829e9c6657efb380debe7

#include "transformer_layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


layer make_transformer_layer(int batch, int h, int w, int c, int history_size, int steps, int train)
{
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.batch = batch;
    l.type = HISTORY;
    l.steps = steps;
    l.history_size = history_size;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_h = h;
    l.out_w = w;
    l.out_c = c * history_size;
    l.inputs = h * w * c;
    l.outputs = h * w * c * history_size;

    l.forward = forward_transformer_layer;
    l.backward = backward_transformer_layer;

    fprintf(stderr, "HISTORY b = %d, s = %2d, steps = %2d   %4d x%4d x%4d -> %4d x%4d x%4d \n", l.batch / l.steps, l.history_size, l.steps, w, h, c, l.out_w, l.out_h, l.out_c);

    l.output = (float*)xcalloc(l.batch * l.outputs, sizeof(float));
    l.delta = (float*)xcalloc(l.batch * l.outputs, sizeof(float));

    l.prev_state_cpu = (float*)xcalloc(l.batch*l.outputs, sizeof(float));

#ifdef GPU

    l.forward_gpu = forward_transformer_layer_gpu;
    l.backward_gpu = backward_transformer_layer_gpu;

    l.output_gpu = cuda_make_array(0, l.batch * l.outputs);
    l.delta_gpu = cuda_make_array(0, l.batch * l.outputs);

    l.prev_state_gpu = cuda_make_array(0, l.batch*l.outputs);

#endif  // GPU

    //l.batch = 4;
    //l.steps = 1;

    return l;
}

void forward_transformer_layer(layer l, network_state state)
{

    float* k = (float*)xcalloc(1, sizeof(layer));

    const int batch = l.batch;
    const int ts = l.steps;
    const int c = l.c;
    const int h = l.h;
    const int w = l.w;
    const int heads = 4;

    const int c_period = c / heads;
    const int num_c_for_each_array = c_period / 3; //for k, q, v

    int idx_k = 0;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < num_c_for_each_array * w * h; i++) {
            for (int head = 0; head < heads; head++) {
                k[idx_k] = state.input[b * c * h * w  + head * c_period * w * h + i];
                idx_k++;

            }

        }

    }

    // [b,c,h,w] -> 3 x [b, heads, hw, c/(3*heads)]

}

void backward_transformer_layer(layer l, network_state state)
{
    if (l.steps == 1) {
        axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
        return;
    }

    const int batch = l.batch / l.steps;

    // l.delta -> state.delta
    int i;
    for (i = 0; i < l.steps; ++i) {
        int b;
        for (b = 0; b < batch; ++b) {
            int input_start = b*l.inputs + i*l.inputs*batch;
            int output_start = b*l.outputs + i*l.outputs*batch;
            float *state_delta = state.delta + input_start;
            float *l_delta = l.delta + output_start;

            //copy_cpu(l.inputs, l_delta, 1, state_delta, 1);
            axpy_cpu(l.inputs, 1, l_delta, 1, state_delta, 1);
        }
    }
}

#ifdef GPU
void forward_transformer_layer_gpu(const layer l, network_state state)
{
    if (l.steps == 1) {
        simple_copy_ongpu(l.inputs*l.batch, state.input, l.output_gpu);
        return;
    }

    const int batch = l.batch / l.steps;

    //int copy_size = l.inputs*batch*l.steps;
    //printf(" copy_size = %d, inputs = %d, batch = %d, steps = %d, l.history_size = %d \n", copy_size, l.inputs, batch, l.steps, l.history_size);
    //simple_copy_ongpu(copy_size, state.input, l.output_gpu);
    //return;

    //fill_ongpu(batch*l.outputs, 0, l.prev_state_gpu, 1);
    float *prev_output = l.prev_state_gpu;

    int i;
    for (i = 0; i < l.steps; ++i) {
        // shift cell
        int shift_size = l.inputs * (l.history_size - 1);
        int output_sift = l.inputs;

        int b;
        for (b = 0; b < batch; ++b) {
            //printf(" hist-fw: i = %d, b = %d \n", i, b);

            int input_start = b*l.inputs + i*l.inputs*batch;
            int output_start = b*l.outputs + i*l.outputs*batch;
            float *input = state.input + input_start;
            float *output = l.output_gpu + output_start;

            //copy_cpu(shift_size, prev_output + b*l.outputs, 1, output + output_sift, 1);
            simple_copy_ongpu(shift_size, prev_output + b*l.outputs, output + output_sift);

            //copy_cpu(l.inputs, input, 1, output, 1);
            simple_copy_ongpu(l.inputs, input, output);

            int h;
            for (h = 1; h < l.history_size; ++h) {
                //scal_ongpu(l.inputs, (l.history_size - h)/ (float)l.history_size, output + h*l.inputs, 1);
                //scal_ongpu(l.inputs, 0, output + h*l.inputs, 1);
            }
        }
        prev_output = l.output_gpu + i*l.outputs*batch;
    }

    int output_start = (l.steps - 1)*l.outputs*batch;
    //copy_cpu(batch*l.outputs, l.output + output_start, 1, l.prev_state_cpu, 1);
    simple_copy_ongpu(batch*l.outputs, l.output_gpu + output_start, l.prev_state_gpu);
}

void backward_transformer_layer_gpu(const layer l, network_state state)
{
    if (l.steps == 1) {
        axpy_ongpu(l.inputs*l.batch, 1, l.delta_gpu, 1, state.delta, 1);
        return;
    }

    const int batch = l.batch / l.steps;

    //int copy_size = l.inputs*batch*l.steps;
    //printf(" copy_size = %d, inputs = %d, batch = %d, steps = %d, l.history_size = %d \n", copy_size, l.inputs, batch, l.steps, l.history_size);
    //axpy_ongpu(copy_size, 1, l.delta_gpu, 1, state.delta, 1);
    //return;

    // l.delta -> state.delta
    int i;
    for (i = 0; i < l.steps; ++i) {
        int b;
        for (b = 0; b < batch; ++b) {
            //printf(" hist-bw: i = %d, b = %d \n", i, b);

            int input_start = b*l.inputs + i*l.inputs*batch;
            int output_start = b*l.outputs + i*l.outputs*batch;
            float *state_delta = state.delta + input_start;
            float *l_delta = l.delta_gpu + output_start;

            //copy_cpu(l.inputs, l_delta, 1, state_delta, 1);
            axpy_ongpu(l.inputs, 1, l_delta, 1, state_delta, 1);
        }
    }
}
#endif