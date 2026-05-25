import serial
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import threading
import joblib
from collections import deque
from pathlib import Path

stop_requested = False


def wait_for_stop_key():
    global stop_requested
    print("\nPress 'y' + Enter to stop acquisition.")
    while True:
        key = input().strip().lower()
        if key == "y":
            stop_requested = True
            print("\nStop requested.")
            break


# ── SensorReader ───────────────────────────────────────────────────────────────

class SensorReader:
    def __init__(self, port, baudrate=115200, n_sensors=2):
        self.port = port
        self.baudrate = baudrate
        self.n_sensors = n_sensors
        self.serial_conn = None
        self.offsets = None
        self.calibrated = False

        self.models = None       # list of fitted sklearn models, one per sensor
        self.zero_fields = None  # list of np.ndarray (3,), one per sensor

        # Unified table: raw B values + predicted forces for both sensors
        self.df = pd.DataFrame(columns=[
            'timestamp_ms',
            'Bx_s1', 'By_s1', 'Bz_s1',
            'Bx_s2', 'By_s2', 'Bz_s2',
            'Fx_s1', 'Fy_s1', 'Fz_s1',
            'Fx_s2', 'Fy_s2', 'Fz_s2',
        ])

        self._prev_ts   = None
        self._prev_data = None  # shape [2, 3]

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            self.serial_conn.reset_input_buffer()
            return True
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return False

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")

    # ── Raw reading ────────────────────────────────────────────────────────────

    def _parse_line(self, line):
        """
        Parse one CSV line into a [2, 3] array.
        Expected format from Arduino: x1,y1,z1,x2,y2,z2
        Returns None on bad input.
        """
        parts = line.strip().split(',')
        if len(parts) != 6:
            return None
        try:
            vals = list(map(float, parts))
        except ValueError:
            return None
        return np.array(vals, dtype=float).reshape(2, 3)

    def read_line(self):
        """Block until one valid line arrives. Returns (timestamp_ms, data[2,3])."""
        while True:
            line = self.serial_conn.readline().decode('utf-8', errors='ignore')
            if not line:
                return None, None
            data = self._parse_line(line)
            if data is not None:
                return time.time() * 1000.0, data

    def read_latest_line(self):
        """Drain the buffer and return only the most recent valid line."""
        latest = None
        while self.serial_conn.in_waiting > 0:
            line = self.serial_conn.readline().decode('utf-8', errors='ignore')
            if line.strip():
                latest = line
        if latest is None:
            return None, None
        data = self._parse_line(latest)
        if data is None:
            return None, None
        return time.time() * 1000.0, data

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_models(self, model_paths):
        """Load one calibration model per sensor from pkl files saved by 02_train_model.py."""
        self.models = []
        self.zero_fields = []
        for i, path in enumerate(model_paths):
            payload = joblib.load(Path(path))
            self.models.append(payload["model"])
            self.zero_fields.append(payload["zero_field"])
            print(f"  Sensor {i+1}: loaded {Path(path).name}  "
                  f"zero_field={payload['zero_field'].round(3)} mT")
        self.offsets = np.stack(self.zero_fields)  # [n_sensors, 3] — used as zero baseline
        self.calibrated = True

    # ── Calibration (re-zero) ──────────────────────────────────────────────────

    def calibrate(self, n_samples=100):
        """Re-capture zero baseline at rest (overrides model zero_field)."""
        print(f"\nRe-zeroing over {n_samples} samples — keep sensors unloaded…")
        samples, collected = [], 0
        while collected < n_samples:
            ts, data = self.read_line()
            if data is not None:
                samples.append(data)
                collected += 1
        self.offsets = np.mean(np.stack(samples, axis=0), axis=0)
        if self.zero_fields is not None:
            self.zero_fields = [self.offsets[i] for i in range(self.n_sensors)]
        self.calibrated = True
        print("Re-zero complete!")
        for i in range(self.n_sensors):
            print(f"  Sensor {i+1}: {self.offsets[i].round(3)} mT")

    # ── Calibrated reading + derivative + logging ──────────────────────────────

    def read_and_log(self):
        """
        Read latest line, apply zero offset, predict forces if models are loaded.
        Appends a row to self.df.

        Returns (timestamp_ms, b_data[2,3], forces[2,3] or None).
        """
        timestamp, data = self.read_latest_line()
        if data is None:
            return None, None, None

        if not self.calibrated:
            print("Warning: not zeroed yet — returning raw data.")

        # Predict forces for each sensor
        forces = np.zeros((self.n_sensors, 3))
        if self.models is not None:
            for i in range(self.n_sensors):
                db = (data[i] - self.zero_fields[i]).reshape(1, -1)
                forces[i] = self.models[i].predict(db).flatten()

        self._prev_ts   = timestamp
        self._prev_data = data

        self.df.loc[len(self.df)] = [
            timestamp,
            data[0, 0], data[0, 1], data[0, 2],
            data[1, 0], data[1, 1], data[1, 2],
            forces[0, 0], forces[0, 1], forces[0, 2],
            forces[1, 0], forces[1, 1], forces[1, 2],
        ]

        return timestamp, data, forces

    # ── Recording loops ────────────────────────────────────────────────────────

    def record_until_stop(self, live_plot=False, max_points=300):
        global stop_requested
        if live_plot:
            plotter = RealtimePlotter(self, max_points=max_points)
            plotter.run_until_stop()
        else:
            print("\nRecording data (no live plot) — press 'y' + Enter to stop.")
            while not stop_requested:
                self.read_and_log()
                time.sleep(0.01)
            print("\nRecording stopped.")

    # ── Persistence ────────────────────────────────────────────────────────────

    def finalize_and_save(self, filename="data_log.csv"):
        print("\nCapture finished.")
        print(self.df.head())
        print("...")
        print(self.df.tail())
        print(f"\nTotal samples: {len(self.df)}")
        self.df.to_csv(filename, index=False)
        print(f"Saved → {filename}")


# ── Real-time plotter ──────────────────────────────────────────────────────────

class RealtimePlotter:
    """
    2-subplot live view of inter-sensor means:
      Top:    |mean_X|, |mean_Y|, |mean_Z|  (force)
      Bottom: |mean_dX/dt|, |mean_dY/dt|, |mean_dZ/dt|  (derivative)

    X mean: sensor1.x is inverted before averaging to account for
            opposite mounting orientation.
    """

    def __init__(self, reader, max_points=300, update_interval_ms=50):
        self.reader = reader
        self.max_points = max_points
        self.update_interval_ms = update_interval_ms
        self._window_open = True

        mk = lambda: deque(maxlen=max_points)
        self.t_buf   = mk()
        self.mx_buf  = mk()   # Fz sensor 1
        self.my_buf  = mk()   # Fz sensor 2
        self.mz_buf  = mk()   # Fx mean
        self.mdx_buf = mk()   # Fy mean

        plt.ion()
        self.fig, (self.ax_force, self.ax_deriv) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True
        )
        self.fig.suptitle("Real-time Sensor Signal — 2-Sensor Mean",
                           fontsize=13, fontweight='bold')
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # ── Top: Fz per sensor ────────────────────────────────────────────────
        self.line_fz1, = self.ax_force.plot([], [], color='b', linewidth=1.5, label='Fz sensor 1')
        self.line_fz2, = self.ax_force.plot([], [], color='r', linewidth=1.5, label='Fz sensor 2')
        self.ax_force.set_title("Normal force Fz per sensor", fontsize=10)
        self.ax_force.set_ylabel("Fz (N)")
        self.ax_force.set_ylim(-2, 20)
        self.ax_force.axhline(0, color='k', linewidth=0.5)
        self.ax_force.legend(loc='upper right')
        self.ax_force.grid(True, alpha=0.3)

        # ── Bottom: Fx and Fy mean ─────────────────────────────────────────────
        self.line_fx, = self.ax_deriv.plot([], [], color='g', linewidth=1.5, label='Fx mean')
        self.line_fy, = self.ax_deriv.plot([], [], color='m', linewidth=1.5, label='Fy mean')
        self.ax_deriv.set_title("Shear forces Fx / Fy (mean across sensors)", fontsize=10)
        self.ax_deriv.set_xlabel("Time (ms)")
        self.ax_deriv.set_ylabel("F (N)")
        self.ax_deriv.set_ylim(-10, 10)
        self.ax_deriv.axhline(0, color='k', linewidth=0.5)
        self.ax_deriv.legend(loc='upper right')
        self.ax_deriv.grid(True, alpha=0.3)

        plt.tight_layout()

    def _on_close(self, event):
        global stop_requested
        self._window_open = False
        stop_requested = True
        print("\nPlot window closed — stopping acquisition.")

    def _read_all_pending(self):
        while self.reader.serial_conn.in_waiting > 0:
            timestamp, data, forces = self.reader.read_and_log()

            if data is not None and forces is not None:
                self.t_buf.append(timestamp)
                self.mx_buf.append(forces[0, 2])               # Fz sensor 1
                self.my_buf.append(forces[1, 2])               # Fz sensor 2
                self.mz_buf.append((forces[0, 0] + forces[1, 0]) / 2)   # Fx mean
                self.mdx_buf.append((forces[0, 1] + forces[1, 1]) / 2)  # Fy mean

    def _redraw(self):
        if not self._window_open:
            return
        try:
            t = list(self.t_buf)
            self.line_fz1.set_data(t, list(self.mx_buf))
            self.line_fz2.set_data(t, list(self.my_buf))
            self.line_fx.set_data(t, list(self.mz_buf))
            self.line_fy.set_data(t, list(self.mdx_buf))

            if len(t) > 1:
                self.ax_force.set_xlim(min(t), max(t))
                self.ax_deriv.set_xlim(min(t), max(t))

            self.fig.canvas.flush_events()
        except Exception:
            self._window_open = False

    def run_until_stop(self):
        global stop_requested
        print("\nRecording with live plot — press 'y' + Enter to stop.")
        last_draw = time.time()

        while not stop_requested and self._window_open:
            self._read_all_pending()
            now = time.time()
            if (now - last_draw) * 1000 >= self.update_interval_ms:
                self._redraw()
                last_draw = now
            time.sleep(0.005)

        print("\nRecording stopped.")
        if self._window_open:
            plt.ioff()
            plt.show(block=False)
            plt.close(self.fig)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filename = input("Output CSV name (no extension): ").strip() or "data_log"
    results_dir = "surface_results"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{filename}.csv")

    LIVE_PLOT = True

    MODEL_S1 = "model_s1.pkl"   # ← path to sensor 1 model
    MODEL_S2 = "model_s2.pkl"   # ← path to sensor 2 model

    PORT = 'COM10'
    reader = SensorReader(port=PORT, baudrate=115200, n_sensors=2)

    if not reader.connect():
        print("Failed to connect.")
        exit(1)

    try:
        print("Loading models…")
        reader.load_models([MODEL_S1, MODEL_S2])
        print("\nKeep sensors unloaded and press Enter to re-zero…")
        input()
        reader.calibrate(n_samples=100)
        print("Move / press the sensors now.")

        keyboard_thread = threading.Thread(target=wait_for_stop_key, daemon=True)
        keyboard_thread.start()

        reader.record_until_stop(live_plot=LIVE_PLOT, max_points=300)
        reader.finalize_and_save(filename=filepath)

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        reader.close()