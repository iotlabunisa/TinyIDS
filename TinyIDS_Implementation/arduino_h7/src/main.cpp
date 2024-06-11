#include <Arduino.h>
#include "../lib/NeuralNetworkPruning/NeuralNetwork.h"
#include "../lib/NeuralNetworkPruning/AccuracyStats.h"
#include "../lib/NeuralNetworkPruning/ArrayUtils.h"

#define BUFFER_SIZE 32

float expectedOutput[OUTPUT_NEURONS] = {0};

float read();

float *run();

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial1.begin(9600);
    while (!Serial1);
    delay(1000);
    randomSeed(analogRead(0));

    initNN();
//    startTrainingNN();
    stopTrainingNN();

    Serial1.println(F("INIT")); //Training Start
    while (1) {
        run();
        if (getEpoch() == 2) {
            stopTrainingNN();
            break;
        }
    }
    Serial1.println(F("INF")); //Training Stop -> Inference Start
}

void loop() {
//    float maxV = 0;//Valore
//    int maxI = getMax(run(), OUTPUT_NEURONS, &maxV);//Indice
//    Serial1.println(maxI);
//    Serial1.flush();
}

/*

















 */
float read() {
    char serialBuff[BUFFER_SIZE];

    for (int i = 0; i < BUFFER_SIZE; i++) {
        while (!Serial1.available()) delay(10);

        serialBuff[i] = (char) Serial1.read();
        if ('\n' == serialBuff[i]) {
            serialBuff[i] = '\0';
            break;
        }
    }

    return atof(serialBuff);
}

void load() {
    int params = isTraining() ? INPUT_NEURONS + OUTPUT_NEURONS : INPUT_NEURONS;
    float *in = getInputArrayNN();

    Serial1.println(F("G"));
    Serial1.println(params);

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
    Serial1.println(F("R"));
    Serial1.flush();
    predictedOutput = runNN(expectedOutput);
    Serial1.println(F("E"));
    Serial1.flush();
    return predictedOutput;
}