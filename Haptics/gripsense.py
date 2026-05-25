#!/usr/bin/env python3
"""
ROS force publisher for one or two TMAG5273 sensors.

This is the ROS-topic version of the prediction loop in 03_run_experiment.py:

    1. Read TMAG5273 serial sample(s): [Bx, By, Bz] in mT.
    2. Subtract each model's zero_field baseline.
    3. Run the corresponding saved sklearn/joblib model.
    4. Publish [Fx, Fy, Fz] on ROS topic(s).

Single-sensor example:
    python3 gripsense.py --port /dev/ttyUSB0 --model model_s1.pkl --rezero
    rostopic echo /gripsense/force

Two-sensor example:
    python3 gripsense.py --two --port /dev/ttyUSB0 --models model_s1.pkl model_s2.pkl --rezero
    rostopic echo /gripsense/force/s1
    rostopic echo /gripsense/force/s2
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from sensors import TMAG5273


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_S1 = SCRIPT_DIR / "model_s1.pkl"
DEFAULT_MODEL_S2 = SCRIPT_DIR / "model_s2.pkl"
FORCE_LABELS = ("Fx_N", "Fy_N", "Fz_N")


def _import_ros():
    try:
        import rospy
        from geometry_msgs.msg import Vector3Stamped
    except ImportError as exc:
        raise SystemExit(
            "Could not import ROS Python packages. Source your ROS setup first, "
            "for example: source /opt/ros/noetic/setup.bash"
        ) from exc
    return rospy, Vector3Stamped


def _resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path

    local_path = SCRIPT_DIR / path
    if local_path.exists():
        return local_path

    return path


def _load_model(path: Path) -> Tuple[Path, dict]:
    model_path = _resolve_model_path(path)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}. Run 02_train_model.py first.")

    payload = joblib.load(model_path)
    missing = [key for key in ("zero_field", "model") if key not in payload]
    if missing:
        sys.exit(f"Model file {model_path} is missing keys: {missing}")

    zero_field = np.asarray(payload["zero_field"], dtype=float).reshape(-1)
    if zero_field.shape != (3,):
        sys.exit(f"Model zero_field must have shape (3,), got {zero_field.shape}")

    if not hasattr(payload["model"], "predict"):
        sys.exit(f"Model object in {model_path} does not provide predict().")

    payload["zero_field"] = zero_field
    return model_path, payload


def _load_models(paths):
    model_paths = []
    model_payloads = []
    for path in paths:
        model_path, payload = _load_model(path)
        model_paths.append(model_path)
        model_payloads.append(payload)
    return model_paths, model_payloads


def _prediction_to_force_xyz(prediction: np.ndarray, axis_labels: Optional[list]) -> np.ndarray:
    force = np.asarray(prediction, dtype=float).reshape(-1)

    if axis_labels is None:
        if force.size < 3:
            raise RuntimeError(f"Model returned {force.size} value(s), expected at least 3.")
        return force[:3]

    if len(axis_labels) != force.size:
        raise RuntimeError(
            f"Model returned {force.size} value(s), but axis_labels has {len(axis_labels)} entries."
        )

    label_to_index = {label: idx for idx, label in enumerate(axis_labels)}
    missing = [label for label in FORCE_LABELS if label not in label_to_index]
    if missing:
        raise RuntimeError(
            f"Model axis_labels must include {list(FORCE_LABELS)} to publish force. "
            f"Missing: {missing}"
        )

    return np.array([force[label_to_index[label]] for label in FORCE_LABELS], dtype=float)


def _estimate_force(field_mT: np.ndarray, model_payload: dict) -> np.ndarray:
    field = np.asarray(field_mT, dtype=float).reshape(3)
    delta_field = (field - model_payload["zero_field"]).reshape(1, -1)
    prediction = model_payload["model"].predict(delta_field)
    return _prediction_to_force_xyz(prediction, model_payload.get("axis_labels"))


def _as_sensor_fields(reading: np.ndarray, n_sensors: int) -> np.ndarray:
    reading = np.asarray(reading, dtype=float)
    if n_sensors == 1:
        return reading.reshape(1, 3)
    return reading.reshape(n_sensors, 3)


def _capture_zero_fields(sensor: TMAG5273, n_samples: int, n_sensors: int):
    samples = [[] for _ in range(n_sensors)]
    for _ in range(n_samples):
        fields = _as_sensor_fields(sensor.read_field_mT(), n_sensors)
        for i in range(n_sensors):
            samples[i].append(fields[i])
    return [np.mean(np.stack(sensor_samples, axis=0), axis=0) for sensor_samples in samples]


def _model_paths_from_args(args, n_sensors: int):
    if args.models is not None:
        if len(args.models) != n_sensors:
            sys.exit(
                f"--two expects exactly 2 models; single-sensor mode expects exactly 1. "
                f"Got {len(args.models)} model(s)."
            )
        return args.models

    if n_sensors == 1:
        return [args.model or DEFAULT_MODEL_S1]

    return [args.model or DEFAULT_MODEL_S1, args.model2 or DEFAULT_MODEL_S2]


def _topic_for_sensor(base_topic: str, sensor_index: int, n_sensors: int) -> str:
    if n_sensors == 1:
        return base_topic
    return f"{base_topic}/s{sensor_index + 1}"


def _frame_id_for_sensor(base_frame_id: str, sensor_index: int, n_sensors: int) -> str:
    if n_sensors == 1:
        return base_frame_id
    return f"{base_frame_id}_s{sensor_index + 1}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read one TMAG5273 sensor, estimate force, and publish it to a ROS topic."
    )
    parser.add_argument("--port", required=True,
                        help="Serial port of the ESP32, e.g. /dev/ttyUSB0 or COM3")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Serial baud rate (default: 115200)")
    parser.add_argument("--two", action="store_true",
                        help="Read two TMAG5273 sensors from one serial line: "
                             "Bx1,By1,Bz1,Bx2,By2,Bz2")
    parser.add_argument("--model", type=Path, default=None,
                        help="Single-sensor model, or sensor 1 model with --two "
                             "(default: model_s1.pkl)")
    parser.add_argument("--model2", type=Path, default=None,
                        help="Sensor 2 model when using --two (default: model_s2.pkl)")
    parser.add_argument("--models", nargs="+", type=Path, default=None,
                        help="Model list overriding --model/--model2. Use one model in "
                             "single mode, or two models with --two.")
    parser.add_argument("--topic", default="/gripsense/force",
                        help="ROS topic for geometry_msgs/Vector3Stamped force. With --two, "
                             "publishes to <topic>/s1 and <topic>/s2 (default: /gripsense/force)")
    parser.add_argument("--frame-id", default="tmag5273",
                        help="Frame id used in the published header (default: tmag5273)")
    parser.add_argument("--node-name", default="gripsense",
                        help="ROS node name (default: gripsense)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Target publish rate in Hz (default: 100)")
    parser.add_argument("--queue-size", type=int, default=10,
                        help="ROS publisher queue size (default: 10)")
    parser.add_argument("--rezero", action="store_true",
                        help="Capture a new zero-field baseline before publishing")
    parser.add_argument("--rezero-samples", type=int, default=200,
                        help="Samples averaged for --rezero (default: 200)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until Ctrl-C)")
    parser.add_argument("--log-throttle", type=float, default=1.0,
                        help="Seconds between force log messages; use 0 to disable (default: 1)")
    return parser


def main():
    args = _build_arg_parser().parse_args()

    if args.rate <= 0:
        sys.exit("--rate must be greater than 0.")
    if args.queue_size <= 0:
        sys.exit("--queue-size must be greater than 0.")
    if args.rezero_samples <= 0:
        sys.exit("--rezero-samples must be greater than 0.")

    rospy, Vector3Stamped = _import_ros()
    n_sensors = 2 if args.two else 1
    model_paths, model_payloads = _load_models(_model_paths_from_args(args, n_sensors))

    rospy.init_node(args.node_name, anonymous=False)
    topics = [_topic_for_sensor(args.topic, i, n_sensors) for i in range(n_sensors)]
    publishers = [
        rospy.Publisher(topic, Vector3Stamped, queue_size=args.queue_size)
        for topic in topics
    ]

    for i, (model_path, payload) in enumerate(zip(model_paths, model_payloads)):
        rospy.loginfo("Sensor %d model: %s", i + 1, model_path)
        rospy.loginfo(
            "Sensor %d zero-field baseline: %s mT",
            i + 1,
            np.round(payload["zero_field"], 6),
        )
    rospy.loginfo(
        "Opening TMAG5273 serial port %s at %d baud for %d sensor(s)",
        args.port,
        args.baud,
        n_sensors,
    )

    sensor = TMAG5273(port=args.port, baud=args.baud, n_sensors=n_sensors)

    try:
        if args.rezero:
            print("Re-zeroing: keep the TMAG5273 sensor(s) unloaded and press Enter.", file=sys.stderr)
            input()
            rospy.loginfo("Collecting %d zero-field samples.", args.rezero_samples)
            zero_fields = _capture_zero_fields(sensor, args.rezero_samples, n_sensors)
            for i, zero_field in enumerate(zero_fields):
                model_payloads[i]["zero_field"] = zero_field
                rospy.loginfo(
                    "Sensor %d new zero-field baseline: %s mT",
                    i + 1,
                    np.round(zero_field, 6),
                )

        rate = rospy.Rate(args.rate)
        started = time.monotonic()
        rospy.loginfo("Publishing geometry_msgs/Vector3Stamped force to %s", ", ".join(topics))

        while not rospy.is_shutdown():
            now = time.monotonic()
            fields_mT = _as_sensor_fields(sensor.read_field_mT(), n_sensors)
            stamp = rospy.Time.now()
            force_log_parts = []

            for i in range(n_sensors):
                force_xyz = _estimate_force(fields_mT[i], model_payloads[i])

                msg = Vector3Stamped()
                msg.header.stamp = stamp
                msg.header.frame_id = _frame_id_for_sensor(args.frame_id, i, n_sensors)
                msg.vector.x = float(force_xyz[0])
                msg.vector.y = float(force_xyz[1])
                msg.vector.z = float(force_xyz[2])
                publishers[i].publish(msg)

                force_log_parts.append(
                    f"s{i + 1}: x={msg.vector.x:.6f} y={msg.vector.y:.6f} z={msg.vector.z:.6f}"
                )

            if args.log_throttle > 0:
                rospy.loginfo_throttle(
                    args.log_throttle,
                    "Force [N]: " + " ; ".join(force_log_parts),
                )

            if args.duration is not None and (now - started) >= args.duration:
                break

            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
        rospy.loginfo("TMAG5273 serial port closed.")


if __name__ == "__main__":
    main()

