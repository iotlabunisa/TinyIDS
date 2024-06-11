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


#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include "Constants.h"

#define HIDDEN_WEIGHT_SIZE ((INPUT_NEURONS + 1) * HIDDEN_LAYER_NEURONS + (HIDDEN_LAYER - 1) * (HIDDEN_LAYER_NEURONS + 1) * HIDDEN_LAYER_NEURONS)
#define OUTPUT_WEIGHT_SIZE ((HIDDEN_LAYER_NEURONS + 1) * OUTPUT_NEURONS)
#define TOTAL_WEIGHTS_SIZE (HIDDEN_WEIGHT_SIZE + OUTPUT_WEIGHT_SIZE)
#define TOTAL_NEURONS      (INPUT_NEURONS + HIDDEN_LAYER_NEURONS * HIDDEN_LAYER + OUTPUT_NEURONS)

#define DELTA_SIZE TOTAL_NEURONS - INPUT_NEURONS
#define ALLOCATED_SIZE     (TOTAL_NEURONS + TOTAL_WEIGHTS_SIZE + TOTAL_NEURONS - INPUT_NEURONS)

#ifdef PRUNING_PERCENT
#define PRUNING_SIZE (int) (TOTAL_WEIGHTS_SIZE * PRUNING_PERCENT)

typedef struct PruningStruct {
    int mask[PRUNING_SIZE] = {0};
    int maskIndex = 0;
    int epochToPrune;
} PruningStruct;
#endif


typedef struct NeuralNetwork {
    float net[ALLOCATED_SIZE] = {0};

#ifdef PRUNING_PERCENT
    PruningStruct pruning;
#endif

    /* All weights (total_weights long). */
    float *weight = net;

    /* Stores input array and output of each neuron (total_neurons long). */
    float *output = weight + TOTAL_WEIGHTS_SIZE;

    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    float *delta = output + TOTAL_NEURONS;

    unsigned short trainning = 1;
    unsigned short loaded = 0;
} NeuralNetwork;

void initNN();

float  *runNN(float  *desiredOutputs);

float *getInputArrayNN();

void startTrainingNN();

void stopTrainingNN();
int isTraining();

void saveNN(char *path);

#endif /*NEURALNETWORK_H*/
