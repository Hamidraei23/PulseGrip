#include "c3dhall11.h"
#include <Wire.h>


// Set pins for I2C
#define I2C1_SDA_PIN 9
#define I2C1_SCL_PIN 8

TwoWire I2Cone = TwoWire(0);

// Sensors setup
const int n_sensors = 1; // Change this when you have more than one sensor
C3dhall11 sensor[n_sensors];
c3dhall11_data_t sensor_data;

volatile bool printDataFlag = false;

hw_timer_t * timer = NULL;
int sampleRate = 100;                       // Hz
uint32_t sampleInterval = 1000000UL / sampleRate;  // microseconds

void IRAM_ATTR timer_isr() {
  printDataFlag = true;
}

void setup() {
  Serial.begin(115200);
  delay(100);

  for (uint8_t i = 0; i < n_sensors; i++)
    sensor[i].setI2CInstance(&I2Cone);

  I2Cone.begin(I2C1_SDA_PIN, I2C1_SCL_PIN, 100000);

  //Configure each sensor
  uint8_t new_address = 0x0A;
  int pins[n_sensors] = {5}; //{5, 6, 7}; // Change also this when you have more than one sensor

  for (int i = 0; i < n_sensors; i++) {
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], HIGH);
    delay(10);

    sensor[i].set_address(new_address);
    sensor[i].default_cfg();

    if (sensor[i].check_communication() == C3DHALL11_OK) {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" connected");
      delay(10);
      //sensor[i].offsetCalibration();
    }
    else {
      Serial.println("error");
    }
    new_address++;
  }

  // Setup timer
  // Setup timer (ESP32 core 3.x API)
  timer = timerBegin(1000000);                 // 1 MHz -> 1 tick = 1 µs
  timerAttachInterrupt(timer, &timer_isr);
  timerAlarm(timer, sampleInterval, true, 0);  // interval in µs, autoreload

}

void loop() {
  if (printDataFlag) {
    printDataFlag = false;

    // Your printing data logic
    //Serial.print("[");
    //Serial.print(millis());
    for (int j = 0; j < n_sensors; j++) {
      if (C3DHALL11_OK == sensor[j].read_data(&sensor_data)) {
        //Serial.print(',');
        Serial.print(sensor_data.x_axis);
        Serial.print(',');
        Serial.print(sensor_data.y_axis);
        Serial.print(',');
        if (j == n_sensors - 1) {
          Serial.println(sensor_data.z_axis);
          //Serial.println("]");
        }
        else {
          Serial.print(sensor_data.z_axis);
          Serial.print(',');
        }
      }
      else {
        if (j == n_sensors - 1) {
          Serial.print(' 0');
          Serial.print(' 0');
          Serial.println(' 0');
        }
        else {
          for (int i = 0; i < 3; i++)
            Serial.print(' 0');
        }
      }
    }
  }
}

/*

#include "c3dhall11.h"
#include <Wire.h>

// I2C pins (ESP32)
#define I2C1_SDA_PIN 21
#define I2C1_SCL_PIN 22

TwoWire I2Cone = TwoWire(0);

// Sensor setup
const int n_sensors = 1;
C3dhall11 sensor[n_sensors];
c3dhall11_data_t sensor_data;

int sampleRate = 100;  // Hz
unsigned long sampleInterval_us; // microseconds between samples
unsigned long lastSampleTime_us = 0;

void setup() {
  Serial.begin(115200);
  delay(100);

  // attach I2C bus to each sensor object
  for (uint8_t i = 0; i < n_sensors; i++) {
    sensor[i].setI2CInstance(&I2Cone);
  }

  // init I2C
  I2Cone.begin(I2C1_SDA_PIN, I2C1_SCL_PIN, 100000); // 100 kHz

  // configure sensors
  uint8_t new_address = 0x0A;
  int pins[n_sensors] = {25}; // enable pins for sensors

  for (int i = 0; i < n_sensors; i++) {
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], HIGH);
    delay(10);

    sensor[i].set_address(new_address);
    sensor[i].default_cfg();

    if (sensor[i].check_communication() == C3DHALL11_OK) {
      Serial.print("Sensor ");
      Serial.print(i + 1);
      Serial.println(" connected");
      delay(10);
      // sensor[i].offsetCalibration();
    } else {
      Serial.println("error");
    }

    new_address++;
  }

  // timing setup
  sampleInterval_us = 1000000UL / sampleRate; // e.g. 10_000 us for 100 Hz
  lastSampleTime_us = micros();
}

void loop() {
  unsigned long now_us = micros();
  if (now_us - lastSampleTime_us >= sampleInterval_us) {
    lastSampleTime_us = now_us;

    // read + print all sensors once per tick
    for (int j = 0; j < n_sensors; j++) {
      if (C3DHALL11_OK == sensor[j].read_data(&sensor_data)) {
        // PRINT IN THE FORMAT MATLAB EXPECTS: x,y,z\n
        Serial.print(sensor_data.x_axis);
        Serial.print(' ');
        Serial.print(sensor_data.y_axis);
        Serial.print(' ');
        Serial.println(sensor_data.z_axis);
      } else {
        // failed read -> print zeros to keep structure
        Serial.print("0,0,0\n");
      }
    }
  }
}

*/