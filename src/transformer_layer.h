#ifndef TRANSFORMER_LAYER_H
#define TRANSFORMER_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

#ifdef __cplusplus
extern "C" {
#endif
layer make_transformer_layer(int batch, int h, int w, int c, int history_size, int steps, int train);
void forward_transformer_layer(layer l, network_state state);
void backward_transformer_layer(layer l, network_state state);

#ifdef GPU
void forward_transformer_layer_gpu(const layer l, network_state state);
void backward_transformer_layer_gpu(const layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif

#endif  // CONV_LSTM_LAYER_H
