

import serial
import time
import re

SERIAL_PORT = "COM5"
BAUD_RATE = 115200

_connection = None


def get_sensor_connection():
    global _connection

    if _connection is None or not _connection.is_open:
        _connection = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        )

        time.sleep(2)

    return _connection


def read_soil_moisture():

    try:
        connection = get_sensor_connection()
        start_time = time.time()

        while time.time() - start_time < 5:

            line = connection.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if "Soil Moisture Raw Value:" in line:
                match = re.search(r"(\d+)$", line)

                if match:
                    return int(match.group(1)), None

        return None, "No sensor reading received."

    except serial.SerialException as error:
        return None, f"Serial connection error: {error}"

    except Exception as error:
        return None, str(error)
