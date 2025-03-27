import sys
import serial
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox,
                               QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QFileDialog)
from PySide6.QtCore import QTimer

class UNIDOSInterface(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UNIDOS E RS232 Interface")
        self.serial_connection = None
        self.measurement_timer = QTimer()
        self.acquired_data = []

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        connection_layout = QHBoxLayout()
        self.port_input = QLineEdit("COM7")
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)

        self.connect_button.clicked.connect(self.connect_serial)
        self.disconnect_button.clicked.connect(self.disconnect_serial)

        connection_layout.addWidget(QLabel("Port:"))
        connection_layout.addWidget(self.port_input)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.disconnect_button)

        self.status_label = QLabel("Status: Disconnected")
        self.state_display = QTextEdit()
        self.state_display.setReadOnly(True)

        measurement_layout = QHBoxLayout()
        self.start_measurement_button = QPushButton("Start Measurement")
        self.stop_measurement_button = QPushButton("Stop Measurement")
        self.stop_measurement_button.setEnabled(False)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Dose", "Dose Rate"])

        self.start_measurement_button.clicked.connect(self.start_measurement)
        self.stop_measurement_button.clicked.connect(self.stop_measurement)

        measurement_layout.addWidget(QLabel("Mode:"))
        measurement_layout.addWidget(self.mode_selector)
        measurement_layout.addWidget(self.start_measurement_button)
        measurement_layout.addWidget(self.stop_measurement_button)

        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)

        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_data)

        self.range_button = QPushButton("Set Measurement Range")
        self.range_button.clicked.connect(self.set_measurement_range)

        self.interval_input = QSpinBox()
        self.interval_input.setRange(1, 1800)
        self.interval_input.setValue(1)
        self.interval_input.setSuffix(" s")
        self.interval_input.valueChanged.connect(self.update_timer_interval)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Acquisition Interval:"))
        interval_layout.addWidget(self.interval_input)

        main_layout.addLayout(connection_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(QLabel("Device State:"))
        main_layout.addWidget(self.state_display)
        main_layout.addLayout(measurement_layout)
        main_layout.addLayout(interval_layout)
        main_layout.addWidget(QLabel("Data Output:"))
        main_layout.addWidget(self.data_display)
        main_layout.addWidget(self.save_button)
        main_layout.addWidget(self.range_button)


        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.measurement_timer.timeout.connect(self.acquire_data)

    def connect_serial(self):
        port = self.port_input.text()
        try:
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            self.status_label.setText(f"Status: Connected to {port}")
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Status: Failed to connect to {port} ({str(e)})")

    def disconnect_serial(self):
        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None

        self.status_label.setText("Status: Disconnected")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

    def send_command(self, command, delay=0.5):
        if not self.serial_connection:
            return "Not connected"

        try:
            full_cmd = command + "\r\n"
            self.serial_connection.write(full_cmd.encode('ascii'))
            self.serial_connection.flush()
            time.sleep(delay)
            response = self.serial_connection.read_all().decode('ascii').strip()
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def start_measurement(self):
        mode = self.mode_selector.currentText()
        command = "M0" if mode == "Dose" else "M1"
        response = self.send_command(command)
        self.state_display.append(f"Mode Set: {mode} -> {response}")

        response = self.send_command("STA")
        self.state_display.append(f"Start Measurement: {response}")

        self.measurement_timer.start(self.interval_input.value() * 1000)
        self.start_measurement_button.setEnabled(False)
        self.stop_measurement_button.setEnabled(True)

    def stop_measurement(self):
        self.measurement_timer.stop()
        response = self.send_command("RES")
        self.state_display.append(f"Stop Measurement: {response}")

        self.start_measurement_button.setEnabled(True)
        self.stop_measurement_button.setEnabled(False)

    def acquire_data(self):
        response = self.send_command("D")
        unit = self.send_command("DU")
        if response.startswith("D"):
            data_parts = response.split(";")
            if len(data_parts) > 3:
                time_elapsed = data_parts[1]
                dose_value = data_parts[5]

                formatted_data = f"Time: {time_elapsed}, Dose: {dose_value}, Unit: {unit[2:]}"
                self.data_display.append(formatted_data)
                self.acquired_data.append(formatted_data)
            else:
                self.data_display.append("Invalid response format")
        else:
            self.data_display.append(response)

    def save_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write("Time, Dose, Unit\n")
                    file.write("\n".join(self.acquired_data))
                self.state_display.append(f"Data saved to {file_path}")
            except Exception as e:
                self.state_display.append(f"Error saving data: {str(e)}")

    def set_measurement_range(self):
        response = self.send_command("RGE")
        self.state_display.append(f"Set Measurement Range: {response}")

    def update_timer_interval(self):
        if self.measurement_timer.isActive():
            self.measurement_timer.setInterval(self.interval_input.value() * 1000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UNIDOSInterface()
    window.show()
    sys.exit(app.exec())
