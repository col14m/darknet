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

float * create_target_vector(layer l, network_state state, int start_idx)
{
    const int batch = l.batch * l.steps;
    const int c = l.c;
    const int h = l.h;
    const int w = l.w;
    const int heads = 4;

    const int num_c_for_each_head = c / heads;
    const int num_c_for_each_vector = num_c_for_each_head / 3; //for k, q, v

    float* target_vector = (float*) xcalloc(sizeof(float), h * w * c * batch / 3);

    int idx_target_vector = 0;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < num_c_for_each_vector * w * h; i++) {
            for (int head = 0; head < heads; head++) {
                target_vector[idx_target_vector] = state.input[b * c * h * w  + head * num_c_for_each_head * w * h + start_idx + i];
                idx_target_vector++;

            }

        }

    }

    return target_vector;

}

void forward_transformer_layer(layer l, network_state state)
{
    const int batch = l.batch * l.steps;
    const int c = l.c;
    const int h = l.h;
    const int w = l.w;
    const int heads = 4;

    const int num_c_for_each_head = c / heads;
    const int num_c_for_each_vector = num_c_for_each_head / 3; //for k, v, q

    // [b,c,h,w] -> 3 x [b, heads, hw, c/(3*heads)]

    float * k_vector = create_target_vector(l, state, 0);
    float * v_vector = create_target_vector(l, state, num_c_for_each_vector * h * w);
    float * q_vector = create_target_vector(l, state, 2 * num_c_for_each_vector * h * w);

    // attn = q x kt
    gemm(0, 1, )

}



void backward_transformer_layer(layer l, network_state state)
{

}

#ifdef GPU
void forward_transformer_layer_gpu(const layer l, network_state state)
{

}

void backward_transformer_layer_gpu(const layer l, network_state state)
{

}
#endif