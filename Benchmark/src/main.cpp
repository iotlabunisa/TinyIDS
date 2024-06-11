#include <SD.h>
#include <string.h>
#include <SoftwareSerial.h>
#include <Adafruit_INA219.h>

#define BUFFER_SIZE 32
#define PIN_SPI_CS 4
#define RX 2
#define TX 3
#define CURRENT_CORRECTION 0.80
#define BATTERY_CAPACITY 800 // 1000mAh -20% per I/O

enum Status {
    INI, LOADING, RUNNING, ERROR
};
const char *StatusToString[] = {"INITIALIZATION", "LOADING DATA", "RUNNING", "ERROR"};

Status s;
File fData, fLog;
Adafruit_INA219 eLog;
SoftwareSerial SerialBridge(RX, TX);
float simulatedBattery = BATTERY_CAPACITY;

void read(char *msg);

float readNum();

void printTime(long time);

void printCurrent(float current, long time);

void setup() {
    pinMode(RX, INPUT);
    pinMode(TX, OUTPUT);
    SerialBridge.begin(9600);
    Serial.begin(9600);

    s = Status::INI;

    if (!eLog.begin()) {
        Serial.println(F("INA219 problem"));
        s = Status::ERROR;
    }

    if (!SD.begin(PIN_SPI_CS) || !SD.exists("data.txt")) {
        Serial.println(F("SD problem"));
        s = Status::ERROR;
    }

    fData = SD.open("data.txt", FILE_READ);
    if (!fData) {
        Serial.println(F("Data problem"));
        s = Status::ERROR;
    }

    Serial.println("FINE SETUP");
}

void loop() {
    Serial.print(F("Status: "));
    Serial.println(StatusToString[s]);
    switch (s) {
        case Status::INI: {
            Serial.println(F("Waiting for device startup"));
            read("INIT");
            s = Status::LOADING;
            break;
        }
        case Status::LOADING: {
            read("G");
            char buffer[BUFFER_SIZE];
            int ioNumber = (int) readNum();
            int len = 0;
            char c;

            do {
                if (len > BUFFER_SIZE) {
                    s = Status::ERROR;
                    Serial.println("BufferOverflow");
                    break;
                }
                c = (char) fData.read();
                if (EOF == c) {
                    fData.close();
                    fData = SD.open("data.txt", FILE_READ);
                    if (!fData) {
                        s = Status::ERROR;
                        Serial.println(F("Data problem"));
                        break;
                    }
                    continue;
                }
                buffer[len++] = c;
                if (',' == c || '\n' == c || '\r' == c || ';' == c) {
                    buffer[--len] = '\0';
                    if (len > 0) {
                        ioNumber--;
                        len = 0;
                        SerialBridge.println(buffer);
                        delay(10);
                    }
                }
            } while (ioNumber >= 0);

            s = Status::RUNNING;
            break;
        }
        case Status::RUNNING: {
            read("R");
            int i = 0;
            float current = 0;
            long time = millis();
            while (1) {
                i++;
                current += eLog.getCurrent_mA() - CURRENT_CORRECTION;
                if (SerialBridge.available() && 'E' == (char) SerialBridge.read()) {
                    time = millis() - time;
                    current = current / i;

                    printTime(time);
                    printCurrent(current, time);
                    break;
                }
            }
            simulatedBattery -= current * ((float) time / 3600);
            Serial.print(F("Totale mAh rimaneneti: "));
            Serial.println(simulatedBattery);
            SerialBridge.println(simulatedBattery);
            s = Status::LOADING;
            break;
        }
        default:
            delay(10000);
    }
}

/*




































*/

void rawRead(char *buffer) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
        while (!SerialBridge.available()) {
            delay(10);
        }

        buffer[i] = (char) SerialBridge.read();
        if ('\n' == buffer[i] || -1 == buffer[i]) {//Condizione -1 non può essere hittata
            buffer[i] = '\0';
            break;
        }
    }
}

void read(char *msg) {
    char buffer[BUFFER_SIZE];
    while (true) {
        rawRead(buffer);
        if (strstr(buffer, msg) != nullptr)
            break;
    }
}

float readNum() {
    char buffer[BUFFER_SIZE];
    rawRead(buffer);
    return atof(buffer);
}


void printTime(long time) {
    uint32_t secs = time / 1000;
    uint32_t ms = time - (secs * 1000);

    Serial.print(F("TaskTime: "));
    Serial.print(secs);
    Serial.print(".");
    if (ms < 100) Serial.print(F("0"));
    if (ms < 10) Serial.print(F("0"));
    Serial.print(ms);
    Serial.println(F("s"));
}


void printCurrent(float current, long time) {
    Serial.print(F("Average Current: "));
    Serial.print(current);
    Serial.println(F("mA"));
    Serial.print(F("Total Current: "));
    Serial.print((current / ((float) time * 1000)) * 60);
    Serial.println(F("mA/min"));
}


/*
int countFiles();
 int countFiles() {
    File directory = SD.open("/log/");
    int count = 0;
    while (true) {
        File entry = directory.openNextFile();
        if (!entry) break;
        count++;
        entry.close();
    }
    return count;
}
    char log[64] = "";
    sprintf(log, "/log/log_%d.txt", countFiles());

    fLog = SD.open(log, FILE_WRITE);
if (!fLog || !fLog.availableForWrite()) {
        Serial.println(F("Log problem"));
}
*/