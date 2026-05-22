"""
Step 1 — Record synchronized data from TMAG5273 and ATI AXIA 80-M20.

Writes a CSV with columns:
    timestamp_s, Bx_mT, By_mT, Bz_mT, Fx_N, Fy_N, Fz_N, Tx_Nm, Ty_Nm, Tz_Nm, phase

phase = "zero"  → sensor unloaded (used by 02_train_model.py to compute zero-field baseline)
phase = "calib" → operator applying forces freely

Usage
-----
    python 01_record_data.py --port COM10 --ati-host 192.168.1.1 --ati-cpf 1000000 --ati-cpt 100000

Add --live-plot to open the real-time force monitor during the calib phase.
Requires matplotlib: pip install matplotlib
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np

from sensors import TMAG5273, ATIAxia80

CSV_COLUMNS = [
    "timestamp_s",
    "Bx_mT", "By_mT", "Bz_mT",
    "Fx_N",  "Fy_N",  "Fz_N",
    "Tx_Nm", "Ty_Nm", "Tz_Nm",
    "phase",
]

# Stability constants — must match 02_train_model.py
_STABILITY_WINDOW = 10    # samples (0.1 s at 100 Hz)
_STABILITY_THR_N  = 0.1   # N rolling std
_HOLD_TARGET_S    = 3.0   # s


# ---------------------------------------------------------------------------
# Live force monitor
# ---------------------------------------------------------------------------

class LiveCalibDisplay:
    """
    Real-time matplotlib window shown during the calib phase.

    Two panels:
      Top  — horizontal bar per axis (Fx, Fy, Fz), centred at zero
      Bottom — hold-timer progress bar (fills when force is stable for 3 s)

    Stability definition matches 02_train_model.py:
      rolling std of |F| over last 10 samples < 0.1 N
    """

    FORCE_RANGE_N = 15.0   # x-axis limits ± this value
    UPDATE_EVERY  = 5      # redraw every N samples (~20 Hz at 100 Hz recording)

    def __init__(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        self._plt = plt

        plt.ion()
        self.fig = plt.figure(figsize=(7, 5))
        self.fig.suptitle("Calibration — Live Force Monitor", fontsize=12, fontweight="bold")

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.5)
        self._ax_f = self.fig.add_subplot(gs[0])
        self._ax_h = self.fig.add_subplot(gs[1])

        self._setup_force_panel()
        self._setup_hold_panel()

        self._history   = collections.deque(maxlen=_STABILITY_WINDOW)
        self._stable_since  = None
        self._n         = 0
        self._open      = True
        self.fig.canvas.mpl_connect("close_event",
                                    lambda _: setattr(self, "_open", False))
        plt.tight_layout()
        plt.show(block=False)
        self.fig.canvas.flush_events()

    def _setup_force_panel(self):
        ax = self._ax_f
        ax.set_xlim(-self.FORCE_RANGE_N, self.FORCE_RANGE_N)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Fx", "Fy", "Fz"], fontsize=13)
        ax.set_xlabel("Force (N)", fontsize=10)
        ax.set_title("Current force per axis", fontsize=10)
        ax.axvline(x=0, color="black", linewidth=1.5, zorder=5)
        ax.grid(axis="x", alpha=0.3)

        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        self._bars = ax.barh([0, 1, 2], [0, 0, 0],
                             color=colors, height=0.55, zorder=4)
        self._flabels = [
            ax.text(0.3, i, "0.00 N", va="center", ha="left",
                    fontsize=10, fontweight="bold")
            for i in [0, 1, 2]
        ]

    def _setup_hold_panel(self):
        ax = self._ax_h
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title("Hold timer  (stable = rolling std < 0.1 N over 0.1 s)",
                     fontsize=9)
        self._hbar = ax.barh([0], [0], color="#3498db", height=0.35)[0]
        self._hlabel = ax.text(0.5, 0, f"0.0 s / {_HOLD_TARGET_S:.0f} s",
                               va="center", ha="center", fontsize=11,
                               fontweight="bold", transform=ax.transAxes)

    def update(self, ft: np.ndarray, t_now: float):
        self._n += 1

        # --- stability tracking -------------------------------------------
        total = float(np.linalg.norm(ft[:3]))
        self._history.append(total)

        if len(self._history) >= _STABILITY_WINDOW:
            std = float(np.std(self._history))
            if std < _STABILITY_THR_N:
                if self._stable_since is None:
                    self._stable_since = t_now
            else:
                self._stable_since = None
        else:
            self._stable_since = None

        hold_s = (t_now - self._stable_since) if self._stable_since else 0.0

        # --- redraw every UPDATE_EVERY samples ----------------------------
        if self._n % self.UPDATE_EVERY != 0 or not self._open:
            return

        # Force bars
        for i, (bar, val, lbl) in enumerate(zip(self._bars, ft[:3], self._flabels)):
            bar.set_width(float(val))
            bar.set_x(min(0.0, float(val)))
            offset = 0.4 if val >= 0 else -0.4
            lbl.set_position((float(val) + offset, i))
            lbl.set_ha("left" if val >= 0 else "right")
            lbl.set_text(f"{val:+.2f} N")

        # Hold bar
        fraction = min(hold_s / _HOLD_TARGET_S, 1.0)
        self._hbar.set_width(fraction)
        if hold_s >= _HOLD_TARGET_S:
            self._hbar.set_color("#2ecc71")
            self._hlabel.set_text(f"HOLD OK   {hold_s:.1f} s")
        else:
            self._hbar.set_color("#3498db")
            self._hlabel.set_text(f"{hold_s:.1f} s / {_HOLD_TARGET_S:.0f} s")

        try:
            self.fig.canvas.flush_events()
        except Exception:
            self._open = False

    def close(self):
        if self._open:
            self._plt.close(self.fig)


# ---------------------------------------------------------------------------
# Recording loop
# ---------------------------------------------------------------------------

def record_phase(
    tmag: TMAG5273,
    ati: ATIAxia80,
    writer: csv.DictWriter,
    phase: str,
    duration_s: float,
    rate_hz: float,
    t_origin: float,
    display: "LiveCalibDisplay | None" = None,
):
    interval = 1.0 / rate_hz
    t_end    = time.monotonic() + duration_s
    n        = 0

    while time.monotonic() < t_end:
        t0 = time.monotonic()
        b  = tmag.read_field_mT()
        ati.drain_buffer()
        ft = ati.read_ft()
        ts = t0 - t_origin

        writer.writerow({
            "timestamp_s": f"{ts:.5f}",
            "Bx_mT": f"{b[0]:.6f}", "By_mT": f"{b[1]:.6f}", "Bz_mT": f"{b[2]:.6f}",
            "Fx_N":  f"{ft[0]:.6f}", "Fy_N":  f"{ft[1]:.6f}", "Fz_N":  f"{ft[2]:.6f}",
            "Tx_Nm": f"{ft[3]:.6f}", "Ty_Nm": f"{ft[4]:.6f}", "Tz_Nm": f"{ft[5]:.6f}",
            "phase": phase,
        })
        n += 1

        if display is not None:
            display.update(ft, t0)
        else:
            remaining = max(0.0, t_end - time.monotonic())
            print(f"  [{phase}] {n} samples  |  {remaining:.1f} s left   ", end="\r")

        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print(f"\n  [{phase}] done: {n} samples")
    return n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record TMAG5273 + ATI data to CSV")

    # ATI
    ati = parser.add_argument_group("ATI AXIA 80-M20")
    ati.add_argument("--ati-host", required=True, help="Sensor IP address")
    ati.add_argument("--ati-port", type=int, default=49152)
    ati.add_argument("--ati-cpf", type=float, default=1_000_000.0,
                     help="Counts per Newton (from ATI calibration file)")
    ati.add_argument("--ati-cpt", type=float, default=1_000_000.0,
                     help="Counts per Newton-metre (from ATI calibration file)")
    ati.add_argument("--tare-samples", type=int, default=200,
                     help="ATI samples averaged for software tare (default 200)")

    # Recording
    rec = parser.add_argument_group("Recording")
    rec.add_argument("--zero-duration", type=float, default=5.0,
                     help="Seconds to record with no force applied (default 5)")
    rec.add_argument("--calib-duration", type=float, default=120.0,
                     help="Seconds to record force application (default 120)")
    rec.add_argument("--rate", type=float, default=100.0,
                     help="Target sampling rate in Hz (default 100)")
    rec.add_argument("--sensor-id", default="s1",
                     help="Sensor identifier used in output filename (default: s1)")
    rec.add_argument("--output", type=Path, default=None,
                     help="Output CSV path (default: calib_<sensor-id>.csv)")
    rec.add_argument("--live-plot", action="store_true",
                     help="Show real-time force monitor during calib phase (requires matplotlib)")

    # TMAG5273
    tmag = parser.add_argument_group("TMAG5273 (via ESP32 serial)")
    tmag.add_argument("--port", required=True,
                      help="Serial port of the ESP32, e.g. COM10 or /dev/ttyUSB0")
    tmag.add_argument("--baud", type=int, default=115200)

    args = parser.parse_args()

    output = args.output or Path(f"calib_{args.sensor_id}.csv")
    if output.exists():
        sys.exit(f"Error: {output} already exists. Rename it or choose a different --sensor-id.")

    print(f"Sensor ID : {args.sensor_id}")
    print(f"Output    : {output}")
    print("Initialising sensors…")
    sensor     = TMAG5273(port=args.port, baud=args.baud)
    ati_sensor = ATIAxia80(
        host=args.ati_host, port=args.ati_port,
        cpf=args.ati_cpf,   cpt=args.ati_cpt,
    )

    try:
        ati_sensor.start_streaming()
        print(f"Taring ATI over {args.tare_samples} samples — keep sensor unloaded…")
        ati_sensor.tare(args.tare_samples)
        print("ATI tared.")

        with open(output, "w", newline="") as fh:
            writer   = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            t_origin = time.monotonic()

            # --- zero phase --------------------------------------------------
            print(f"\nZero phase ({args.zero_duration:.0f} s): keep sensor unloaded.")
            print("Press Enter to start zero recording…")
            input()
            ati_sensor.drain_buffer()   # discard packets buffered while waiting
            record_phase(sensor, ati_sensor, writer, "zero",
                         args.zero_duration, args.rate, t_origin)

            # --- calib phase -------------------------------------------------
            print(f"\nCalib phase ({args.calib_duration:.0f} s):")
            print("Apply forces manually — cover normal (Fz), shear (Fx, Fy) and combinations.")
            print("Hold each force level still for at least 3 s.")
            print("Press Enter to start recording…")
            input()
            ati_sensor.drain_buffer()   # discard packets buffered while waiting

            display = LiveCalibDisplay() if args.live_plot else None
            try:
                record_phase(sensor, ati_sensor, writer, "calib",
                             args.calib_duration, args.rate, t_origin,
                             display=display)
            finally:
                if display is not None:
                    display.close()

        print(f"\nSaved: {output.resolve()}")

    finally:
        ati_sensor.close()
        sensor.close()


if __name__ == "__main__":
    main()
