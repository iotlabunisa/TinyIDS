//
// Created by Gennaro on 02/02/2024.
//

#ifndef SIMPLEFFNN_STATS_H
#define SIMPLEFFNN_STATS_H

#include "Constants.h"
#include <float.h>
#include <math.h>

typedef float (*lossFunction)( float,  float);

void initAccuracyStats();

float getLoss();
int getAccuracyStatTotal();
int getEpoch();
void incrementEpoch();

void updateError( float *predictedOutput,  float *desiredOutput, lossFunction fun);

inline float meanSquaredError( float predictedOutput,  float desiredOutput) {
     float error = desiredOutput - predictedOutput;
    return error * error;
}

inline float crossEntropy( float predictedOutput,  float desiredOutput) {
    return -1 * desiredOutput * log(predictedOutput);
}

inline float linearError( float predictedOutput,  float desiredOutput) {
    return desiredOutput - predictedOutput;
}

#endif //SIMPLEFFNN_STATS_H

/*





















int getHigherIndex( float *arr,  int len) {
    int higherIndex = -1;
    float max = FLT_MIN;
    for (int i = 0; i < len; i++) {
        if (arr[i] > max) {
            max = arr[i];
            higherIndex = i;
        }
    }
    return higherIndex;
}
 */