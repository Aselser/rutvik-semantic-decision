import serial

class PulseSender:
    def __init__(self, arduino_port, baud_rate):
        # Connection to the Arduino board via serial communication
        self._serial = serial.Serial(arduino_port, baud_rate)


    def _send_pulse_to_arduino(self):
        # Message to send to Arduino
        message = "P"
        # Send the message to Arduino
        self._serial.write(message.encode())