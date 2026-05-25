"""
Step 2 — Train the force calibration model from recorded CSV data.

Reads the CSV written by 01_record_data.py, computes the zero-field baseline
from the "zero" phase rows, trains a degree-2 polynomial + linear regression
model on the "calib" phase rows, and reports per-axis train/test R² and RMSE.

Usage
-----
    python 02_train_model.py --input calibration_data.csv

Optional: choose which ATI channels to predict (default: Fx, Fy, Fz)
    python 02_train_model.py --input data.csv --output-axes 0,1,2,3,4,5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Maps axis index → (CSV column name, physical unit)
_ATI_AXES = {
    0: ("Fx_N",  "N"),
    1: ("Fy_N",  "N"),
    2: ("Fz_N",  "N"),
    3: ("Tx_Nm", "Nm"),
    4: ("Ty_Nm", "Nm"),
    5: ("Tz_Nm", "Nm"),
}
_B_COLS = ["Bx_mT", "By_mT", "Bz_mT"]


def build_model(model_type: str, degree: int, random_state: int):
    if model_type == "poly":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg",  LinearRegression()),
        ])
    elif model_type == "ridge":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg",  Ridge(alpha=1.0)),
        ])
    elif model_type == "random-forest":
        return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_type == "gradient-boost":
        return MultiOutputRegressor(HistGradientBoostingRegressor(random_state=random_state))
    else:
        raise ValueError(f"Unknown --model-type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train force calibration model from CSV")
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV file written by 01_record_data.py")
    parser.add_argument("--output-axes", default="0,1,2",
                        help="ATI channel indices to predict, comma-separated. "
                             "0=Fx 1=Fy 2=Fz 3=Tx 4=Ty 5=Tz  (default: 0,1,2)")
    parser.add_argument("--model-type", default="poly",
                        choices=["poly", "ridge", "random-forest", "gradient-boost"],
                        help="Model type: poly (default), ridge, random-forest, gradient-boost")
    parser.add_argument("--poly-degree", type=int, default=4,
                        help="Polynomial degree — only used for poly and ridge (default 2)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction held out for test evaluation (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stability-window", type=int, default=10,
                        help="Rolling window (samples) for stability filter (default 10 = 0.1 s at 100 Hz)")
    parser.add_argument("--stability-threshold", type=float, default=0.1,
                        help="Max rolling std of ATI force magnitude to keep a sample (default 0.1 N)")
    parser.add_argument("--sensor-id", default=None,
                        help="Sensor identifier, e.g. s1, s2. If set, default model "
                             "output becomes model_<sensor-id>.pkl")
    parser.add_argument("--model-output", type=Path, default=None,
                        help="Where to save the fitted model (default: model_<sensor-id>.pkl)")
    args = parser.parse_args()

    # Resolve model output path
    if args.model_output is not None:
        model_output = args.model_output
    elif args.sensor_id is not None:
        model_output = Path(f"model_{args.sensor_id}.pkl")
    else:
        model_output = Path("calibration_model.pkl")

    if model_output.exists():
        sys.exit(f"Error: {model_output} already exists. "
                 "Rename it or choose a different --sensor-id / --model-output.")

    output_axes = [int(x) for x in args.output_axes.split(",")]
    for ax in output_axes:
        if ax not in _ATI_AXES:
            raise ValueError(f"Invalid axis index {ax}. Must be 0–5.")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.input}…")
    df = pd.read_csv(args.input)

    required = _B_COLS + [col for ax in output_axes for col, _ in [_ATI_AXES[ax]]] + ["phase"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    zero_rows  = df[df["phase"] == "zero"]
    calib_rows = df[df["phase"] == "calib"]

    if len(zero_rows) == 0:
        raise ValueError("No rows with phase='zero' found. Cannot compute zero-field baseline.")
    if len(calib_rows) < 10:
        raise ValueError(f"Only {len(calib_rows)} calib rows — need at least 10 to fit a model.")

    print(f"  zero rows : {len(zero_rows)}")
    print(f"  calib rows: {len(calib_rows)} (before stability filter)")

    # ------------------------------------------------------------------
    # Zero-field baseline
    # ------------------------------------------------------------------
    zero_field = zero_rows[_B_COLS].values.mean(axis=0)
    print(f"\nZero-field baseline (mT): Bx={zero_field[0]:.4f}  "
          f"By={zero_field[1]:.4f}  Bz={zero_field[2]:.4f}")

    # ------------------------------------------------------------------
    # Stability filter — discard transition samples between force steps
    # ------------------------------------------------------------------
    force_cols   = ["Fx_N", "Fy_N", "Fz_N"]
    force_mag    = np.linalg.norm(calib_rows[force_cols].values.astype(float), axis=1)
    rolling_std  = (pd.Series(force_mag)
                    .rolling(window=args.stability_window, center=True)
                    .std()
                    .fillna(args.stability_threshold + 1))   # edges → filtered out
    stable_mask  = rolling_std.values < args.stability_threshold
    calib_stable = calib_rows[stable_mask]

    print(f"  stable rows: {stable_mask.sum()} / {len(calib_rows)}  "
          f"(window={args.stability_window} samples, threshold={args.stability_threshold} N)")

    if len(calib_stable) < 10:
        raise ValueError(
            f"Only {len(calib_stable)} rows survive the stability filter. "
            "Try raising --stability-threshold or lowering --stability-window."
        )

    # ------------------------------------------------------------------
    # Build feature matrix X (ΔB) and target matrix Y
    # ------------------------------------------------------------------
    B     = calib_stable[_B_COLS].values.astype(float)
    X     = B - zero_field                                # (N, 3) ΔB in mT

    col_names = [_ATI_AXES[ax][0] for ax in output_axes]
    units     = [_ATI_AXES[ax][1] for ax in output_axes]
    Y         = calib_stable[col_names].values.astype(float)  # (N, k)

    # ------------------------------------------------------------------
    # Train / test split and fit
    # ------------------------------------------------------------------
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.random_state
    )
    if args.model_type in ("poly", "ridge"):
        n_feat = PolynomialFeatures(args.poly_degree, include_bias=False).fit(X[:1]).n_output_features_
        print(f"\nSplit: {len(X_tr)} train / {len(X_te)} test  "
              f"(model={args.model_type}, degree={args.poly_degree}, poly features={n_feat})")
    else:
        print(f"\nSplit: {len(X_tr)} train / {len(X_te)} test  (model={args.model_type})")

    model = build_model(args.model_type, args.poly_degree, args.random_state)
    model.fit(X_tr, Y_tr)

    Y_pred_tr = model.predict(X_tr)
    Y_pred_te = model.predict(X_te)

    # Per-axis metrics
    print(f"\n{'Axis':>6s}  {'train R²':>10s}  {'test R²':>10s}  "
          f"{'train RMSE':>14s}  {'test RMSE':>14s}")
    print("-" * 62)
    for k, (lbl, unit) in enumerate(zip(col_names, units)):
        r2_tr   = r2_score(Y_tr[:, k], Y_pred_tr[:, k])
        r2_te   = r2_score(Y_te[:, k], Y_pred_te[:, k])
        rmse_tr = np.sqrt(mean_squared_error(Y_tr[:, k], Y_pred_tr[:, k]))
        rmse_te = np.sqrt(mean_squared_error(Y_te[:, k], Y_pred_te[:, k]))
        print(f"  {lbl:>6s}  {r2_tr:>10.4f}  {r2_te:>10.4f}  "
              f"{rmse_tr:>10.4f} {unit:>3s}  {rmse_te:>10.4f} {unit:>3s}")

    # Refit on full stable dataset before saving
    model.fit(X, Y)
    print(f"\nFull-data R² (all axes combined): {model.score(X, Y):.4f}  "
          f"[{len(X)} stable samples]")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    payload = {
        "zero_field":   zero_field,      # np.ndarray (3,) in mT
        "model":        model,           # fitted sklearn Pipeline
        "output_axes":  output_axes,     # list[int]
        "axis_labels":  col_names,       # list[str]  e.g. ["Fx_N", "Fy_N", "Fz_N"]
        "axis_units":   units,           # list[str]
    }
    joblib.dump(payload, model_output)
    print(f"\nModel saved → {model_output.resolve()}")


if __name__ == "__main__":
    main()
