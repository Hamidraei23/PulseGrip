"""
Step 3 — Real-time force estimation from one or more TMAG5273 sensors.

Loads one calibration model per sensor (saved by 02_train_model.py) and
streams calibrated force vectors. No ATI needed during experiments.

Single sensor
-------------
    python 03_run_experiment.py --port COM3 --models model_s1.pkl --rezero

Two sensors
-----------
    python 03_run_experiment.py --port COM3 --n-sensors 2 \
        --models model_s1.pkl model_s2.pkl --rezero

Output CSV columns (2 sensors, Fx/Fy/Fz per sensor):
    timestamp_s,
    Bx_s1_mT, By_s1_mT, Bz_s1_mT, Fx_s1_N, Fy_s1_N, Fz_s1_N,
    Bx_s2_mT, By_s2_mT, Bz_s2_mT, Fx_s2_N, Fy_s2_N, Fz_s2_N

Output file is auto-named experiment_YYYYMMDD_HHMMSS.csv so it never
overwrites a previous session. Override with --output.

Re-zeroing
----------
Always use --rezero between subjects. The resting magnetic field shifts with
different ring fits and skin deformation. Each sensor is re-zeroed independently.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

from sensors import TMAG5273


def _load_model(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Model not found: {path}. Run 02_train_model.py first.")
    return joblib.load(path)


def _make_output_path(n_sensors: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"experiment_{ts}.csv")


def _build_csv_columns(n_sensors: int, models: list[dict]) -> list[str]:
    cols = ["timestamp_s"]
    for i, m in enumerate(models):
        sid = i + 1
        cols += [f"Bx_s{sid}_mT", f"By_s{sid}_mT", f"Bz_s{sid}_mT"]
        cols += [f"{lbl}_s{sid}" for lbl in m.get("axis_labels", ["Fx_N", "Fy_N", "Fz_N"])]
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="Real-time TMAG5273 → force estimation (single or multi-sensor)"
    )
    parser.add_argument("--models", nargs="+", type=Path, required=True,
                        help="Calibration model(s) from 02_train_model.py, "
                             "one per sensor in order: model_s1.pkl model_s2.pkl ...")
    parser.add_argument("--n-sensors", type=int, default=None,
                        help="Number of sensors (default: inferred from --models)")
    parser.add_argument("--rezero", action="store_true",
                        help="Re-capture zero-field baseline before streaming (recommended)")
    parser.add_argument("--rezero-samples", type=int, default=200,
                        help="Samples averaged for re-zero per sensor (default 200)")
    parser.add_argument("--rate",     type=float, default=100.0,
                        help="Target sampling rate in Hz (default 100)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until Ctrl-C)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: experiment_YYYYMMDD_HHMMSS.csv)")

    hw = parser.add_argument_group("TMAG5273 (via ESP32 serial)")
    hw.add_argument("--port", required=True,
                    help="Serial port of the ESP32, e.g. COM3 or /dev/ttyUSB0")
    hw.add_argument("--baud", type=int, default=115200)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve n_sensors
    # ------------------------------------------------------------------
    n_sensors = args.n_sensors or len(args.models)
    if n_sensors != len(args.models):
        sys.exit(f"--n-sensors {n_sensors} but {len(args.models)} model(s) provided. "
                 "Provide exactly one model per sensor.")

    # ------------------------------------------------------------------
    # Load calibration models
    # ------------------------------------------------------------------
    models = [_load_model(p) for p in args.models]
    zero_fields = [m["zero_field"].copy() for m in models]   # one (3,) per sensor

    print(f"# {n_sensors} sensor(s) loaded:", file=sys.stderr)
    for i, (p, m) in enumerate(zip(args.models, models)):
        print(f"#   Sensor {i+1}: {p.name}  outputs={m.get('axis_labels')}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Output file — never overwrite
    # ------------------------------------------------------------------
    output = args.output or _make_output_path(n_sensors)
    if output.exists():
        sys.exit(f"Output file {output} already exists. Delete it or let auto-naming pick a new name.")

    # ------------------------------------------------------------------
    # Initialise sensor(s) — one serial port, n_sensors channels
    # ------------------------------------------------------------------
    sensor = TMAG5273(port=args.port, baud=args.baud, n_sensors=n_sensors)

    # ------------------------------------------------------------------
    # Re-zero each sensor independently
    # ------------------------------------------------------------------
    if args.rezero:
        print("# Re-zeroing — keep ALL sensors unloaded and press Enter…", file=sys.stderr)
        input()
        print(f"# Collecting {args.rezero_samples} samples per sensor…", file=sys.stderr)
        samples_list = [[] for _ in range(n_sensors)]
        for _ in range(args.rezero_samples):
            reading = sensor.read_field_mT()          # (3,) or (n,3)
            if n_sensors == 1:
                samples_list[0].append(reading)
            else:
                for i in range(n_sensors):
                    samples_list[i].append(reading[i])
        for i in range(n_sensors):
            zero_fields[i] = np.array(samples_list[i]).mean(axis=0)
            print(f"#   Sensor {i+1} zero-field: {zero_fields[i]} mT", file=sys.stderr)
    else:
        for i in range(n_sensors):
            print(f"# Sensor {i+1} using saved zero-field: {zero_fields[i]} mT", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build CSV structure
    # ------------------------------------------------------------------
    csv_columns = _build_csv_columns(n_sensors, models)

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------
    interval = 1.0 / args.rate
    t_start  = time.monotonic()

    print(f"# Output → {output}", file=sys.stderr)
    print("# Streaming — press Ctrl-C to stop.", file=sys.stderr)

    try:
        with open(output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_columns)
            writer.writeheader()

            while True:
                t0      = time.monotonic()
                reading = sensor.read_field_mT()     # (3,) or (n, 3)
                elapsed = t0 - t_start

                row = {"timestamp_s": f"{elapsed:.5f}"}

                for i in range(n_sensors):
                    sid = i + 1
                    b   = reading if n_sensors == 1 else reading[i]
                    db  = (b - zero_fields[i]).reshape(1, -1)
                    f   = models[i]["model"].predict(db).flatten()
                    lbls = models[i].get("axis_labels", ["Fx_N", "Fy_N", "Fz_N"])

                    row[f"Bx_s{sid}_mT"] = f"{b[0]:.6f}"
                    row[f"By_s{sid}_mT"] = f"{b[1]:.6f}"
                    row[f"Bz_s{sid}_mT"] = f"{b[2]:.6f}"
                    for lbl, val in zip(lbls, f):
                        row[f"{lbl}_s{sid}"] = f"{val:.6f}"

                writer.writerow(row)

                if args.duration is not None and elapsed >= args.duration:
                    break

                sleep_t = interval - (time.monotonic() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n# Stopped by user.", file=sys.stderr)
    finally:
        sensor.close()
        print(f"# Saved → {output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
