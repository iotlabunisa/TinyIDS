#include <Arduino.h>
#include "../lib/NeuralNetworkPruning/NeuralNetwork.h"
#include "../lib/NeuralNetworkPruning/AccuracyStats.h"

#define BUFFER_SIZE 32

float expectedOutput[OUTPUT_NEURONS] = {0};

float read();

float *run();

void setup() {
    Serial.begin(9600);
    while (!Serial);
    delay(1000);

    initNN();
//    startTrainingNN();
    stopTrainingNN();

    Serial.println(F("\nINIT")); //Training Start
    while (1) {
        run();
        if (getEpoch() == 2) {
            stopTrainingNN();
            break;
        }
    }
    Serial.println(F("INF")); //Training Stop -> Inference Start
}

void loop() {
//    float maxV = 0;//Valore
//    int maxI = getMax(run(), OUTPUT_NEURONS, &maxV);//Indice
//    Serial.println(maxI);
//    Serial.flush();
}

/*

















 */
float read() {
    char serialBuff[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        while (!Serial.available()) delay(10);

        serialBuff[i] = (char) Serial.read();
        if (10 == (int)serialBuff[i] || 0 == (int)serialBuff[i]) {
            serialBuff[i] = '\0';
            break;
        }
    }
    return atof(serialBuff);
}

void load() {
    int params = isTraining() ? INPUT_NEURONS + OUTPUT_NEURONS : INPUT_NEURONS;
    float *in = getInputArrayNN();

    Serial.println(F("G"));
    Serial.println(params);

    for (int j = 0; j < INPUT_NEURONS; ++j) {
        in[j] = read();
    }

    if (params != INPUT_NEURONS) {
        for (int j = 0; j < OUTPUT_NEURONS; ++j) {
            expectedOutput[j] = read();
        }
    }
}

float *run() {
    load();
    float *predictedOutput;
    Serial.println(F("R"));
    Serial.flush();
    predictedOutput = runNN(expectedOutput);
    Serial.println(F("E"));
    Serial.flush();
    return predictedOutput;
}