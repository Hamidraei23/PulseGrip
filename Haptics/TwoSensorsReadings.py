import serial
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import threading
from collections import deque

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

        # Unified table: raw values + derivatives for both sensors
        self.df = pd.DataFrame(columns=[
            'timestamp_ms',
            'x1', 'y1', 'z1',
            'x2', 'y2', 'z2',
            'dx1', 'dy1', 'dz1',
            'dx2', 'dy2', 'dz2',
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

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate(self, n_samples=10):
        """Collect n_samples at rest and compute per-sensor zero offsets."""
        print(f"\nStarting calibration with {n_samples} samples...")
        print("Please ensure the sensors are not being touched!")
        samples, collected = [], 0
        while collected < n_samples:
            ts, data = self.read_line()
            if data is not None:
                samples.append(data)
                collected += 1
                print(f"  Sample {collected}/{n_samples}")
        # offsets shape: [2, 3]
        self.offsets = np.mean(np.stack(samples, axis=0), axis=0)
        self.calibrated = True
        print("\nCalibration complete!")
        for i in range(self.n_sensors):
            print(f"  Sensor {i+1}: X={self.offsets[i,0]:.2f}  "
                  f"Y={self.offsets[i,1]:.2f}  Z={self.offsets[i,2]:.2f}")

    # ── Calibrated reading + derivative + logging ──────────────────────────────

    def read_and_log(self):
        """
        Read latest line, apply calibration, compute finite-difference derivatives.
        Appends a row to self.df.

        Returns (timestamp_ms, calibrated_data[2,3], deriv[2,3] or None).
        """
        timestamp, data = self.read_latest_line()
        if data is None:
            return None, None, None

        if not self.calibrated:
            print("Warning: not calibrated yet — returning raw data.")
        else:
            data = data - self.offsets  # [2, 3]

        deriv = None
        if self._prev_ts is not None:
            dt = timestamp - self._prev_ts
            if dt > 0:
                deriv = (data - self._prev_data) / dt  # [2, 3]

        self._prev_ts   = timestamp
        self._prev_data = data

        x1, y1, z1 = data[0]
        x2, y2, z2 = data[1]

        if deriv is not None:
            dx1, dy1, dz1 = deriv[0]
            dx2, dy2, dz2 = deriv[1]
        else:
            dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = np.nan

        self.df.loc[len(self.df)] = [
            timestamp,
            x1, y1, z1,
            x2, y2, z2,
            dx1, dy1, dz1,
            dx2, dy2, dz2,
        ]

        return timestamp, data, deriv

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
        self.mx_buf  = mk()   # mean X (with inversion)
        self.my_buf  = mk()   # mean Y
        self.mz_buf  = mk()   # mean Z
        self.mdx_buf = mk()   # mean |dX/dt|
        self.mdy_buf = mk()   # mean |dY/dt|
        self.mdz_buf = mk()   # mean |dZ/dt|

        plt.ion()
        self.fig, (self.ax_force, self.ax_deriv) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True
        )
        self.fig.suptitle("Real-time Sensor Signal — 2-Sensor Mean",
                           fontsize=13, fontweight='bold')
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # ── Top: mean force ────────────────────────────────────────────────────
        self.line_x, = self.ax_force.plot([], [], color='r', linewidth=1.5, label='|mean X|')
        self.line_y, = self.ax_force.plot([], [], color='g', linewidth=1.5, label='|mean Y|')
        self.line_z, = self.ax_force.plot([], [], color='b', linewidth=1.5, label='|mean Z|')
        self.ax_force.set_title("Mean |Force| across sensors  (X: sensor 1 inverted)",
                                fontsize=10)
        self.ax_force.set_ylabel("|Force| (units)")
        self.ax_force.set_ylim(0, 15)        # ← adjust
        self.ax_force.legend(loc='upper right')
        self.ax_force.grid(True, alpha=0.3)

        # ── Bottom: mean derivative ────────────────────────────────────────────
        self.line_dx, = self.ax_deriv.plot([], [], color='r', linewidth=1.5, label='|mean dX/dt|')
        self.line_dy, = self.ax_deriv.plot([], [], color='g', linewidth=1.5, label='|mean dY/dt|')
        self.line_dz, = self.ax_deriv.plot([], [], color='b', linewidth=1.5, label='|mean dZ/dt|')
        self.ax_deriv.set_title("Mean |Derivative| across sensors", fontsize=10)
        self.ax_deriv.set_xlabel("Time (ms)")
        self.ax_deriv.set_ylabel("dV/dt (units/ms)")
        self.ax_deriv.set_ylim(0, 0.2)      # ← adjust
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
            timestamp, data, deriv = self.reader.read_and_log()

            if data is not None:
                # Invert sensor 1 X before averaging (flip if needed)
                x_mean = abs((-data[0, 0] + data[1, 0]) / 2)
                y_mean = abs((data[0, 1]  + data[1, 1]) / 2)
                z_mean = abs((data[0, 2]  + data[1, 2]) / 2)
                self.t_buf.append(timestamp)
                self.mx_buf.append(x_mean)
                self.my_buf.append(y_mean)
                self.mz_buf.append(z_mean)

            if deriv is not None:
                dx_mean = (abs(deriv[0, 0]) + abs(deriv[1, 0])) / 2
                dy_mean = (abs(deriv[0, 1]) + abs(deriv[1, 1])) / 2
                dz_mean = (abs(deriv[0, 2]) + abs(deriv[1, 2])) / 2
                self.mdx_buf.append(dx_mean)
                self.mdy_buf.append(dy_mean)
                self.mdz_buf.append(dz_mean)

    def _redraw(self):
        if not self._window_open:
            return
        try:
            t = self.t_buf

            self.line_x.set_data(t, self.mx_buf)
            self.line_y.set_data(t, self.my_buf)
            self.line_z.set_data(t, self.mz_buf)

            n = min(len(t), len(self.mdx_buf))
            t_trunc = list(t)[-n:]
            self.line_dx.set_data(t_trunc, self.mdx_buf)
            self.line_dy.set_data(t_trunc, self.mdy_buf)
            self.line_dz.set_data(t_trunc, self.mdz_buf)

            if len(t) > 1:
                self.ax_force.set_xlim(min(t), max(t))

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

    PORT = 'COM10'
    reader = SensorReader(port=PORT, baudrate=115200, n_sensors=2)

    if not reader.connect():
        print("Failed to connect.")
        exit(1)

    try:
        reader.calibrate(n_samples=10)
        print("Move / press the sensors now.")

        keyboard_thread = threading.Thread(target=wait_for_stop_key, daemon=True)
        keyboard_thread.start()

        reader.record_until_stop(live_plot=LIVE_PLOT, max_points=300)
        reader.finalize_and_save(filename=filepath)

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        reader.close()