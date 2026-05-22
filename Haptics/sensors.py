"""
Hardware drivers shared across the three pipeline scripts.

Classes
-------
TMAG5273   — 3-axis Hall-effect sensor via ESP32 serial (pyserial)
ATIAxia80  — 6-DoF force/torque sensor over NetFT/UDP
"""

import socket
import serial
import struct
import time
import warnings

import numpy as np


class SaturationWarning(UserWarning):
    """Emitted when a TMAG5273 axis reading is at or near full scale."""


class TMAG5273:
    """
    Reads TMAG5273 magnetic field data from an ESP32 over USB serial.

    The ESP32 firmware (serial_comm_isr, single-sensor mode) prints one
    line per timer tick at 100 Hz:

        Bx_mT,By_mT,Bz_mT\\n

    Values are already converted to mT by the Arduino driver using the
    sensitivity constants from c3dhall11.cpp:
        X/Y range ±40 mT  → sensitivity 820 counts/mT
        Z   range ±80 mT  → sensitivity 410 counts/mT  (default_cfg)

    Parameters
    ----------
    port    : serial port name, e.g. "COM3" (Windows) or "/dev/ttyUSB0" (Linux)
    baud    : baud rate — must match Serial.begin() in the Arduino sketch (115200)
    timeout : readline timeout in seconds
    """

    # Saturation limits match default_cfg in c3dhall11.cpp
    _SAT_XY_MT = 39.9   # mT  (±40 mT range)
    _SAT_Z_MT  = 79.9   # mT  (±80 mT range)

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1,
                 n_sensors: int = 1):
        """
        n_sensors : number of sensors sharing this serial port.
                    The Arduino must be configured with the same n_sensors value.
                    n_sensors=1  →  read_field_mT() returns shape (3,)
                    n_sensors>1  →  read_field_mT() returns shape (n_sensors, 3)
        """
        self._n   = n_sensors
        self._ser = serial.Serial(port, baud, timeout=timeout)
        self._flush_startup()

    def _flush_startup(self, duration_s: float = 0.5):
        """Discard startup messages printed by Arduino setup() (e.g. 'Sensor 1 connected')."""
        t_end = time.monotonic() + duration_s
        while time.monotonic() < t_end:
            self._ser.readline()

    def _parse_serial_line(self, raw: str):
        """Parse a comma-separated line. Returns list of floats or None if invalid."""
        parts = raw.split(",")
        if len(parts) != 3 * self._n:
            return None
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None

    def read_field_mT(self) -> np.ndarray:
        """
        Drain the serial buffer and return the LATEST valid reading.
        Matches OneSensorReading.py's read_latest_line() approach — avoids
        returning stale data that accumulated while the loop was busy elsewhere.

        Single sensor  (n_sensors=1): returns shape (3,)
        Multiple sensors (n_sensors>1): returns shape (n, 3)
        Emits SaturationWarning for any axis at or near its range limit.
        """
        latest = None

        # Drain all currently buffered lines, keep the last valid one
        while self._ser.in_waiting > 0:
            raw    = self._ser.readline().decode("ascii", errors="ignore").strip()
            parsed = self._parse_serial_line(raw)
            if parsed is not None:
                latest = parsed

        # If buffer was empty or had no valid line, block for the next fresh line
        if latest is None:
            while True:
                raw    = self._ser.readline().decode("ascii", errors="ignore").strip()
                parsed = self._parse_serial_line(raw)
                if parsed is not None:
                    latest = parsed
                    break

        arr = np.array(latest, dtype=float).reshape(self._n, 3)

        for i in range(self._n):
            sat = []
            if abs(arr[i, 0]) >= self._SAT_XY_MT: sat.append("X")
            if abs(arr[i, 1]) >= self._SAT_XY_MT: sat.append("Y")
            if abs(arr[i, 2]) >= self._SAT_Z_MT:  sat.append("Z")
            if sat:
                warnings.warn(
                    f"Sensor {i+1} TMAG5273 axis {sat} saturated — check range setting.",
                    SaturationWarning, stacklevel=2,
                )

        return arr[0] if self._n == 1 else arr

    def close(self):
        self._ser.close()


# ---------------------------------------------------------------------------
# ATI AXIA 80-M20 — NetFT/UDP driver
# ---------------------------------------------------------------------------

class ATIAxia80:
    """
    UDP NetFT driver for the ATI AXIA 80-M20 6-DoF force/torque sensor.

    Parameters
    ----------
    host    : IP address of the sensor (e.g. "192.168.1.1")
    port    : NetFT UDP port (default 49152)
    cpf     : counts per Newton   — from the ATI .cal file or web interface
    cpt     : counts per Nm       — from the ATI .cal file or web interface
    timeout : UDP receive timeout in seconds

    Notes
    -----
    cpf / cpt are calibration-specific. Read them from the sensor web UI at
    http://<host>/ → Configuration → Calibration, or from the shipped .cal file.
    The placeholder default of 1 000 000 must be replaced with your values.
    """

    _HEADER    = 0x1234
    _CMD_STOP  = 0x0000
    _CMD_START = 0x0002
    # 36-byte response: RDT seq (I) + FT seq (I) + status (I) + 6×counts (i)
    _FMT  = "!IIIiiiiii"
    _SIZE = struct.calcsize(_FMT)   # 36 bytes

    def __init__(
        self,
        host: str,
        port: int = 49152,
        cpf: float = 1_000_000.0,
        cpt: float = 1_000_000.0,
        timeout: float = 5.0,
    ):
        self._cpf   = cpf
        self._cpt   = cpt
        self._scale = np.array([1/cpf, 1/cpf, 1/cpf, 1/cpt, 1/cpt, 1/cpt])
        self._bias  = np.zeros(6)
        self._sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)
        self._sock.connect((host, port))

    def _send(self, command: int, sample_count: int = 0):
        self._sock.send(struct.pack("!HHI", self._HEADER, command, sample_count))

    def _recv_counts(self) -> np.ndarray:
        data = self._sock.recv(self._SIZE)
        _, _, status, fx, fy, fz, tx, ty, tz = struct.unpack(self._FMT, data)
        if status != 0:
            warnings.warn(f"ATI NetFT status: 0x{status:08X}", RuntimeWarning, stacklevel=3)
        return np.array([fx, fy, fz, tx, ty, tz], dtype=float)

    def start_streaming(self):
        """Start continuous streaming (infinite packets)."""
        self._send(self._CMD_START, 0)
        time.sleep(0.5)   # give the ATI time to start sending packets

    def stop_streaming(self):
        self._send(self._CMD_STOP)

    def tare(self, n_samples: int = 200):
        """Software tare: capture mean bias over n_samples at rest."""
        counts = np.array([self._recv_counts() for _ in range(n_samples)])
        self._bias = counts.mean(axis=0)

    def drain_buffer(self):
        """
        Discard all stale packets queued in the UDP socket buffer.
        Call this just before starting a recording phase to ensure
        read_ft() returns current data, not packets buffered while waiting.
        """
        self._sock.setblocking(False)
        try:
            while True:
                self._sock.recv(self._SIZE)
        except BlockingIOError:
            pass
        finally:
            self._sock.setblocking(True)

    def read_ft(self) -> np.ndarray:
        """Return [Fx, Fy, Fz, Tx, Ty, Tz] in [N, N, N, Nm, Nm, Nm]."""
        return -(self._recv_counts() - self._bias) * self._scale

    def close(self):
        try:
            self.stop_streaming()
        finally:
            self._sock.close()
