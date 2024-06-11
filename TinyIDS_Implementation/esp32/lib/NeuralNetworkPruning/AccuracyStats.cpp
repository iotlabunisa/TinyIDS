//
// Created by Gennaro on 02/02/2024.
//

#include "AccuracyStats.h"
#include <assert.h>

typedef struct AccuracyStats {
    int totals;
    float errorSum;
    int epoch;
} AccuracyStats;

AccuracyStats aStats;

void initAccuracyStats() {
    aStats.totals = 0;
    aStats.errorSum = 0;
    aStats.epoch = 1;
}

float getLoss() {
    return aStats.errorSum / aStats.totals;
}

int getAccuracyStatTotal() {
    return aStats.totals;
}

int getEpoch() {
    return aStats.epoch;
}

void incrementEpoch() {
    aStats.epoch++;
}

void updateError( float *predictedOutput,  float *desiredOutput, lossFunction fun) {
    aStats.totals++;
    if (aStats.totals % BATCH_SIZE == 0) incrementEpoch();
    if (desiredOutput != nullptr) {
        float loss = 0.0;
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            loss += fun(predictedOutput[i], desiredOutput[i]);
        }
        aStats.errorSum += loss;
    }
}