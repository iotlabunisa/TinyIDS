/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "NeuralNetwork.h"
#include "AccuracyStats.h"
#include "ArrayUtils.h"
#include "ESP8266TrueRandom.h"

#include <stdio.h>
#include <math.h>
#include <assert.h>

using namespace std;

float sigmoid(float a);

float derSigmoid(float a);

float *feedForward();

void backPropagate(float *desiredOutputs, lossFunction loss);

void sdgUpdate();

void adamUpdate();

void prune();

NeuralNetwork NN;


void debug() {
#ifdef DEBUG
    float netCopy[ALLOCATED_SIZE];
    for (int copy = 0; copy < TOTAL_NEURONS + TOTAL_WEIGHTS_SIZE + TOTAL_NEURONS - INPUT_NEURONS; copy++)
        netCopy[copy] = NN.net[copy];

    int x = TOTAL_WEIGHTS_SIZE;
    int y = TOTAL_NEURONS;
    int z = TOTAL_NEURONS - INPUT_NEURONS;
    float *d = NN.delta;
    float *w = NN.weight;

    for (int i = 0; i < TOTAL_WEIGHTS_SIZE; i++) {
        netCopy[i] = 0;
    }
    for (int i = 0; i < TOTAL_NEURONS; i++) {
        netCopy[TOTAL_WEIGHTS_SIZE + i] = 1;
    }
    for (int i = 0; i < TOTAL_NEURONS - INPUT_NEURONS; i++) {
        netCopy[TOTAL_WEIGHTS_SIZE + TOTAL_NEURONS + i] = 2;
    }
#endif
    asm("nop");
}

void initNN() {
    for (int i = 0; i < TOTAL_WEIGHTS_SIZE; i++) {
        NN.weight[i] = ((float) ESP8266TrueRandom.random() / (float) RAND_MAX) - 0.5f;
    }

    for (int i = 0; i < PRUNING_SIZE; ++i) {
        NN.pruning.mask[i] = -1;
    }
    NN.pruning.epochToPrune =  EPOCH / 300;
}

float *runNN(float *desiredOutputs) {
    assert(NN.loaded == 1);
    NN.loaded = 0;
    float *predictedOutput = feedForward();
    if (NN.trainning && desiredOutputs != nullptr) {
        backPropagate(desiredOutputs, linearError);
#ifdef ADAM
        adamUpdate();
#else
        sdgUpdate();
#endif

#ifdef PRUNING_PERCENT
        if (getEpoch() % NN.pruning.epochToPrune == NN.pruning.epochToPrune - 1 &&
            NN.pruning.maskIndex < PRUNING_SIZE) {
            prune();
            NN.pruning.epochToPrune += (2 << NN.pruning.maskIndex) *  EPOCH / 300;
        }
#endif

#ifdef REGRESSION
        updateError(predictedOutput, desiredOutputs, meanSquaredError);
#else
        updateError(predictedOutput, desiredOutputs, crossEntropy);
#endif
    }
    return predictedOutput;
}

void stopTrainingNN() {
    NN.trainning = 0;
    initAccuracyStats();
}

void startTrainingNN() {
    NN.trainning = 1;
    initAccuracyStats();
}

float *getInputArrayNN() {
    NN.loaded = 1;
    return NN.output;
}

int isTraining() {
    return NN.trainning;
}

void saveNN(char *path) {
    FILE *f = fopen(path, "w");
    if (f != nullptr) {
        float *w = NN.weight;
        for (int i = 0; i < TOTAL_WEIGHTS_SIZE; ++i) {
            fprintf(f, "%f, ", *w++);
        }
    }
    fclose(f);
}

void loadNN(char *path) {
    FILE *f = fopen(path, "r");
    if (f != nullptr) {
        int i;
        for (i = 0; i < TOTAL_WEIGHTS_SIZE; i++) {
            fscanf(f, "%f, ", &NN.weight[i]);
        }
        while (i++ < ALLOCATED_SIZE) {
            NN.weight[i] = 0;
        }
    }
    fclose(f);
}

/*


















































*/
float *feedForward() {
//     float *w = NN.weight;
    float *o = NN.output + INPUT_NEURONS;
    float *i = NN.output;

    int h, j, k;
    int wIndex = 0;

    /* Figure input layer */
    for (j = 0; j < HIDDEN_LAYER_NEURONS; ++j, o++) {
        float sum = in(NN.pruning.mask, PRUNING_SIZE, wIndex++) ? 0 : NN.weight[wIndex - 1] * BIAS;
        for (k = 0; k < INPUT_NEURONS; ++k) {
            sum += in(NN.pruning.mask, PRUNING_SIZE, wIndex++) ? 0 : NN.weight[wIndex - 1] * i[k];
        }
        *o = sigmoid(sum);
    }

    i += INPUT_NEURONS;

    /* Figure hidden layers, if any. */
    for (h = 1; h < HIDDEN_LAYER; ++h) {
        for (j = 0; j < HIDDEN_LAYER_NEURONS; ++j, o++) {
            float sum = in(NN.pruning.mask, PRUNING_SIZE, wIndex++) ? 0 : NN.weight[wIndex - 1] * BIAS;
            for (k = 0; k < HIDDEN_LAYER_NEURONS; ++k) {
                sum += in(NN.pruning.mask, PRUNING_SIZE, wIndex++) ? 0 : NN.weight[wIndex - 1] * i[k];
            }
            *o = sigmoid(sum);
        }

        i += HIDDEN_LAYER_NEURONS;
    }

    float *ret = o;

    /* Figure output layer. */
    for (j = 0; j < OUTPUT_NEURONS; ++j, o++) {
        float sum = in(NN.pruning.mask, PRUNING_SIZE, wIndex++) ? 0 : NN.weight[wIndex - 1] * BIAS;
        for (k = 0; k < HIDDEN_LAYER_NEURONS; ++k) {
            sum += in(NN.pruning.mask, PRUNING_SIZE, wIndex++) ? 0 : NN.weight[wIndex - 1] * i[k];
        }
        *o = sigmoid(sum);
    }


    return ret;
}

void backPropagate(float *desiredOutputs, lossFunction loss) {
    int h, j, k;

    /* First set the output layer deltas. */
    {
        float *o = NN.output + INPUT_NEURONS + HIDDEN_LAYER_NEURONS * HIDDEN_LAYER; /* First output. */
        float *d = NN.delta + HIDDEN_LAYER_NEURONS * HIDDEN_LAYER; /* First delta. */
        float *t = desiredOutputs; /* First desired output. */

        /* Set output layer deltas. */
        for (j = 0; j < OUTPUT_NEURONS; j++, d++, o++, t++) {
            *d = loss(*o, *t) * derSigmoid(*o);
        }
    }

    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = HIDDEN_LAYER - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        float *o = NN.output + INPUT_NEURONS + (h * HIDDEN_LAYER_NEURONS);
        float *d = NN.delta + (h * HIDDEN_LAYER_NEURONS);

        /* Find first delta in following layer (which may be hidden or output). */
        float *dd = NN.delta + ((h + 1) * HIDDEN_LAYER_NEURONS);

        /* Find first weight in following layer (which may be hidden or output). */
        float *ww =
                NN.weight + ((INPUT_NEURONS + 1) * HIDDEN_LAYER_NEURONS) +
                ((HIDDEN_LAYER_NEURONS + 1) * HIDDEN_LAYER_NEURONS * (h));

        for (j = 0; j < HIDDEN_LAYER_NEURONS; ++j, ++o, ++d) {

            float delta = 0;

            for (k = 0; k < (h == HIDDEN_LAYER - 1 ? OUTPUT_NEURONS : HIDDEN_LAYER_NEURONS); ++k) {
                float forward_delta = dd[k];
                int windex = k * (HIDDEN_LAYER_NEURONS + 1) + (j + 1);
                float forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }

            *d = derSigmoid(*o) * delta;
        }
    }
}

void sdgUpdate() {
    int h, j, k;

    /* Train the outputs. */
    {
        /* Find first output delta. */
        float *d = NN.delta + HIDDEN_LAYER_NEURONS * HIDDEN_LAYER; /* First output delta. */

        /* Find first weight to first output delta. */
        float *w = NN.weight + (HIDDEN_LAYER
                                ? ((INPUT_NEURONS + 1) * HIDDEN_LAYER_NEURONS +
                                   (HIDDEN_LAYER_NEURONS + 1) * HIDDEN_LAYER_NEURONS * (HIDDEN_LAYER - 1))
                                : (0));

        /* Find first output in previous layer. */
        float *i = NN.output + (HIDDEN_LAYER
                                ? (INPUT_NEURONS + (HIDDEN_LAYER_NEURONS) * (HIDDEN_LAYER - 1))
                                : 0);

        /* Set output layer weights. */
        for (j = 0; j < OUTPUT_NEURONS; ++j, ++d) {
            *w++ += *d * LEARNING_RATE * BIAS;
            for (k = 1; k < (HIDDEN_LAYER ? HIDDEN_LAYER_NEURONS : INPUT_NEURONS) + 1; ++k) {
                *w++ += *d * LEARNING_RATE * i[k - 1];
            }
        }
    }


    /* Train the hidden layers. */
    for (h = HIDDEN_LAYER - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        float *d = NN.delta + (h * HIDDEN_LAYER_NEURONS);

        /* Find first input to this layer. */
        float *i = NN.output + (h
                                ? (INPUT_NEURONS + HIDDEN_LAYER_NEURONS * (h - 1))
                                : 0);

        /* Find first weight to this layer. */
        float *w = NN.weight + (h
                                ? ((INPUT_NEURONS + 1) * HIDDEN_LAYER_NEURONS +
                                   (HIDDEN_LAYER_NEURONS + 1) * (HIDDEN_LAYER_NEURONS) * (h - 1))
                                : 0);


        for (j = 0; j < HIDDEN_LAYER_NEURONS; ++j, ++d) {
            *w++ += *d * LEARNING_RATE * BIAS;
            for (k = 1; k < (h == 0 ? INPUT_NEURONS : HIDDEN_LAYER_NEURONS) + 1; ++k) {
                *w++ += *d * LEARNING_RATE * i[k - 1];
            }
        }
    }
}

#ifdef PRUNING_PERCENT

void prune() {
    float min = FLT_MAX;
    int minIndex = -1;

    for (int i = 0; i < PRUNING_SIZE; ++i) {
        if (notIn(NN.pruning.mask, PRUNING_SIZE, i) && NN.weight[i] < min) {
            min = fabs(NN.weight[i]);
            minIndex = i;
        }
    }

    if (minIndex != -1) {
        NN.pruning.mask[NN.pruning.maskIndex++] = minIndex;
//        printf("Pruning %d | Epohch %d\n", minIndex, getEpoch());
    }
}

#endif


void adamUpdate() {
//    printf("TODO");
    // m = beta1 * m + (1-beta1) * g
    // v = beta2 * v + (1-beta2) * g*g
    // lrt = lr * sqrt(1-beta2^t) / (1-beta1)^t
    // params = params - lrt * m / (sqrt(v) + eps)

//    float v[DELTA_SIZE] = {0};
//    float m[DELTA_SIZE] = {0};
//    Adam Params
//    float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
//    -------------------//
//
//    {
//        float  *d = NN.delta;
//        for (int i = 0; i < DELTA_SIZE; i++, d++) {
//            m[i] = beta1 * m[i] + (1 - beta1) * (*d);
//            v[i] = beta2 * v[i] + (1 - beta2) * ((*d) * (*d));
//        }
//    }
//
//    {
//        float *w = NN.weight;
//        for (int h = 0; h < HIDDEN_LAYER + 1; h++) {
//            for (int i = 0; i < (h == 0 ? INPUT_NEURONS : HIDDEN_LAYER_NEURONS) + 1; ++i) {
//                for (int k = 0; k < (h == HIDDEN_LAYER ? OUTPUT_NEURONS : HIDDEN_LAYER_NEURONS); ++k) {
//                    *w++ += LEARNING_RATE * m[HIDDEN_LAYER_NEURONS * h + k] /
//                            (sqrt(v[HIDDEN_LAYER_NEURONS * h + k]) + epsilon);
//                }
//            }
//        }
//    }
}

float sigmoid(float a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0f / (1 + expf(-a));
}

float derSigmoid(float a) {
    if (a == -45.0) return 0;
    if (a == 45.0) return 1;
    return a * (1.0f - a);
}
