
import serial
import time

esp32 = serial.Serial('COM5', 115200, timeout=2)

time.sleep(2)

print("ESP32 Connected Successfully!")

while True:
    reading = esp32.readline().decode('utf-8', errors='ignore').strip()

    if reading:
        print(reading)
