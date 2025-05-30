
#ifndef EML_NET_H
#define EML_NET_H

#include "eml_common.h"
#include "eml_net_common.h"

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO: implement elu
// TODO: implement SeLu for SNN
// TODO: implement HardSigmoid


/** @struct EmlNetLayer
*  Layer of a Neural Network
*
* \internal
*/
typedef struct _EmlNetLayer {
    int32_t n_outputs;
    int32_t n_inputs;
    const float *weights;
    const float *biases;
    EmlNetActivationFunction activation;
} EmlNetLayer;

/** @typedef EmlNet
* \brief Neural Network
*
* Handle used to do inference.
* Normally the initialization code is generated by emlearn.
*/
typedef struct _EmlNet {
    // Layers of the neural network
    int32_t n_layers;
    const EmlNetLayer *layers;
    // Buffers for storing activations
    float *activations1;
    float *activations2;
    int32_t activations_length;
} EmlNet;


static float
eml_net_relu(float in) {
    return (in <= 0.0f) ? 0.0f : in; 
}

static float
eml_net_expit(float in) {
    return 1.0f / (1.0f + expf(-in));
}

static float
eml_net_tanh(float in) {
    return tanhf(in); 
}

static EmlError
eml_net_softmax(float *input, size_t input_length)
{
    EML_PRECONDITION(input, EmlUninitialized);

    float input_max = -INFINITY;
    for (size_t i = 0; i < input_length; i++) {
        if (input[i] > input_max) {
            input_max = input[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < input_length; i++) {
        sum += expf(input[i] - input_max);
    }

    const float offset = input_max + logf(sum);
    for (size_t i = 0; i < input_length; i++) {
        input[i] = expf(input[i] - offset);
    }

    return EmlOk;
}

int32_t
eml_net_argmax(const float *values, int32_t values_length) {

    float vmax = -INFINITY;
    int32_t argmax = -1;
    for (int i=0; i<values_length; i++) {
        if (values[i] > vmax) {
            vmax = values[i];
            argmax = i;
        }
    }
    return argmax;
}


static bool
eml_net_valid(EmlNet *model) {
    bool not_null = model->layers && model->activations1 && model->activations2;
    return not_null;
}

static inline int32_t
eml_net_outputs(EmlNet *model) {
    return model->layers[model->n_layers-1].n_outputs;
}

// For binary problem, one output, we need to report [ prob(class_0), prob(class_1)]
static inline int32_t
eml_net_outputs_proba(EmlNet *model) {
    int32_t n_outputs = eml_net_outputs(model);
    if (n_outputs == 1) {
        n_outputs = 2;
    }
    return n_outputs;
}

/*
* \internal
* \brief Calculate size of activation value arrays
* 
*/
static int32_t
eml_net_find_largest_layer(EmlNet *model) {
    int32_t largest = -1;
    for (int i=0; i<model->n_layers; i++) {
        if (model->layers[i].n_inputs > largest) {
            largest = model->layers[i].n_inputs;
        }
        if (model->layers[i].n_outputs > largest) {
            largest = model->layers[i].n_outputs;
        }
    }
    return largest;
}


// CMSIS-NN tricks
// - fixed-point math
// - quantize to 8 or 16 bit (q7,q15)
// - block-based matrix and vector multiplication
// - weight reordering to match multiplication blocks
// CNN
// - partial im2col, reordering image to match convolution kernel (and size).
// - split x-y pooling. in-place. 4.5x speedup
// Activations
// - SIMD relu on signbit
// CIFAR-10, 80% accuracy, 100ms. 87KB weights, 55KB activations 

// MobileNets
// depthwise-separable convolution, for multiple (color) channels
//
// Strassen matrix multiplication
// Winograd filter-based convolution, 16mul instead of 36. 2-3x speedup on GPU
// "Fast algorithms for convolutional neural networks"
// https://arxiv.org/abs/1509.09308
//
// Convolutional Kernel Networks
// https://papers.nips.cc/paper/5348-convolutional-kernel-networks.pdf
// Approximation of CNN with Gaussian kernels, on unsupervised feature kernels
// reached state-of-art in MINST/CIFAR-10 with linear SVM classifier
// scattering transform also did well

// Inference for a single layer
EmlError
eml_net_forward(const float *in, int32_t in_length,
                const float *weights,
                const float *biases,
                EmlNetActivationFunction activation,
                float *out, int32_t out_length)
{

    // multiply inputs by weights
    for (int o=0; o<out_length; o++) {
        float sum = 0.0f;
        for (int i=0; i<in_length; i++) {
            const int w_idx = o+(i*out_length);
            const float w = weights[w_idx];
            sum += w * in[i];
        }
        out[o] = sum + biases[o];
    }

    // apply activation function
    if (activation == EmlNetActivationIdentity) {
        // no-op
    } else if (activation == EmlNetActivationRelu) {
        for (int i=0; i<out_length; i++) {
            out[i] = eml_net_relu(out[i]);
        }
    } else if (activation == EmlNetActivationLogistic) {
        for (int i=0; i<out_length; i++) {
            out[i] = eml_net_expit(out[i]);
        }

    } else if (activation == EmlNetActivationTanh) {
        for (int i=0; i<out_length; i++) {
            out[i] = eml_net_tanh(out[i]);
        }

    } else if (activation == EmlNetActivationSoftmax) {
        eml_net_softmax(out, out_length);

    } else {
        return EmlUnsupported;
    }

    return EmlOk;
}


EmlError
eml_net_layer_forward(const EmlNetLayer *layer,
                    const float *in, int32_t in_length,
                    float *out, int32_t out_length)
{
    EML_PRECONDITION(in_length >= layer->n_inputs, EmlSizeMismatch);
    EML_PRECONDITION(out_length >= layer->n_outputs, EmlSizeMismatch);
    EML_PRECONDITION(layer->weights, EmlUninitialized);
    EML_PRECONDITION(layer->biases, EmlUninitialized);

    const EmlError err = eml_net_forward(in, layer->n_inputs,
            layer->weights,
            layer->biases,
            layer->activation,
            out, layer->n_outputs
    );

    return err;
}


/*
* \internal
* \brief Run inference
* 
* Used internally by eml_net_predict et.c.
* NOTE: Leaves results in activations2
*/
EmlError
eml_net_infer(EmlNet *model, const float *features, int32_t features_length)
{
    EML_PRECONDITION(eml_net_valid(model), EmlUninitialized);
    EML_PRECONDITION(model->n_layers >= 2, EmlUnsupported);
    EML_PRECONDITION(features_length == model->layers[0].n_inputs, EmlSizeMismatch);
    EML_PRECONDITION(model->activations_length >= eml_net_find_largest_layer(model), EmlSizeMismatch);

    const int32_t buffer_length = model->activations_length; 
    float *buffer1 = model->activations1;
    float *buffer2 = model->activations2;

    // Input layer
    EML_CHECK_ERROR(eml_net_layer_forward(&model->layers[0], features,
                        features_length, buffer1, buffer_length));

    // Hidden layers
    for (int l=1; l<model->n_layers-1; l++) {
        const EmlNetLayer *layer = &model->layers[l];
        // PERF: avoid copying, swap buffers instead
        EML_CHECK_ERROR(eml_net_layer_forward(layer, buffer1, buffer_length, buffer2, buffer_length));
        for (int i=0; i<buffer_length; i++) {
            buffer1[i] = buffer2[i];
        }
    }

    // Output layer
    EML_CHECK_ERROR(eml_net_layer_forward(&model->layers[model->n_layers-1],
                        buffer1, buffer_length, buffer2, buffer_length));

    return EmlOk;
}

/**
* \brief Run inference and return probabilities. Sum of outputs must be approx. 1.
*
* \param model EmlNet instance
* \param features Input data values
* \param features_length Length of input data
* \param out Buffer to store output
* \param out_length Length of output buffer
*
* \return EmlOk on success, else an error
*/
EmlError
eml_net_predict_proba(EmlNet *model, const float *features, int32_t features_length,
                                  float *out, int32_t out_length)
{
    EML_PRECONDITION(eml_net_valid(model), EmlUninitialized);
    EML_PRECONDITION(features, EmlUninitialized);
    EML_PRECONDITION(out, EmlUninitialized);
    const int32_t n_outputs = eml_net_outputs_proba(model);
    EML_PRECONDITION(out_length == n_outputs, EmlSizeMismatch);

    EML_CHECK_ERROR(eml_net_infer(model, features, features_length));

    float proba_sum = 0.0f;

    if (n_outputs == 2) {
        out[1] = model->activations2[0];
        out[0] = 1.0f - out[1];
        proba_sum = out[0] + out[1];
    } else {
        for (int i=0; i<n_outputs; i++) {
            const float p = model->activations2[i];
            out[i] = p;
            proba_sum += p; 
        }
    }

    EML_POSTCONDITION(fabsf(proba_sum - 1.0f) < 0.001f, EmlPostconditionFailed);

    return EmlOk;
}


/**
* \brief Run inference and return most probable class
*
* \param model EmlNet instance
* \param features Input data values
* \param features_length Length of input data
*
* \return The class number, or -EmlError on failure
*/
int32_t
eml_net_predict(EmlNet *model, const float *features, int32_t features_length) {

    const EmlError error = eml_net_infer(model, features, features_length);
    if (error != EmlOk) {
        return -error;
    }

    const int32_t n_outputs = eml_net_outputs(model);

    int32_t _class = -EmlUnknownError;
    if (n_outputs == 1) {
        _class = (model->activations2[0] > 0.5f) ? 1 : 0;
    } else if (n_outputs > 1) {
        _class = eml_net_argmax(model->activations2, n_outputs);
    }

    return _class;
}

/**
* \brief Run inference and return the predicted float array (last layer).
*
* \param model EmlNet instance
* \param features Input data values
* \param features_length Length of input data
* \param out Buffer to store output
* \param out_length Length of output buffer
*
* \return EmlOk on success, or error on failure
*/
EmlError
eml_net_regress(EmlNet *model, const float *features, int32_t features_length, float *out, int32_t out_length)
{
    EML_PRECONDITION(out, EmlUninitialized);
    const int32_t n_outputs = eml_net_outputs(model);
    EML_PRECONDITION(out_length == n_outputs, EmlSizeMismatch);
    EML_CHECK_ERROR(eml_net_infer(model, features, features_length));

    for (int i = 0; i < n_outputs; i++)
    {
        const float p = model->activations2[i];
        out[i] = p;
    }

    return EmlOk;
}

/**
 * \brief Run inference and return single regression value
 *
 * \param model EmlNet instance
 * \param features Input data values
 * \param features_length Length of input data
 *
 * \return The output value on success, or NAN on failure
 */
float
eml_net_regress1(EmlNet *model, const float *features, int32_t features_length)
{
    float out[1];
    EmlError err = eml_net_regress(model, features, features_length, out, 1);
    if (err != EmlOk)
    {
        return NAN;
    }
    return out[0];
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // EML_NET_H
