"""
validate_offline.py  -  Step 3 / Gate 1  (ECE 486 Lab 2)
=========================================================
Offline validation of the forward kinematics model against the provided
Lab2DesignData.txt file.

HOW TO RUN (at home, no robot needed):
    python validate_offline.py

The data file must be in the same folder as this script.
Output is printed to the console and saved to validate_offline_results.txt.

Data file format (500 rows, space-separated):
    x  y  z  J1  J2  pos6
    (x, y, z in mm;  J1, J2, pos6 in degrees)
"""

import numpy as np
import sys
import os
from fk import forward_kinematics, pos6_to_j3, L1, L2, L3

DATA_FILE = 'Lab2DesignData.txt'
OUT_FILE  = 'validate_offline_results.txt'


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.\n"
            f"Place Lab2DesignData.txt in the same folder as this script."
        )
    return np.loadtxt(path)   # shape (500, 6)


def run_validation(data, verbose=True):
    x_true   = data[:, 0]
    y_true   = data[:, 1]
    z_true   = data[:, 2]
    J1_deg   = data[:, 3]
    J2_deg   = data[:, 4]
    pos6_deg = data[:, 5]
    n = len(data)

    # Apply FK to every row
    x_pred = np.zeros(n)
    y_pred = np.zeros(n)
    z_pred = np.zeros(n)
    for i in range(n):
        x_pred[i], y_pred[i], z_pred[i] = forward_kinematics(
            J1_deg[i], J2_deg[i], pos6_deg[i]
        )

    # Errors
    err_x  = x_pred - x_true
    err_y  = y_pred - y_true
    err_z  = z_pred - z_true
    err_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    lines = []
    def out(s=""):
        lines.append(s)
        if verbose:
            print(s)

    out("=" * 65)
    out("OFFLINE FK VALIDATION  --  Lab2DesignData.txt")
    out("=" * 65)
    out(f"Link lengths used :  L1={L1} mm   L2={L2} mm   L3={L3} mm")
    out(f"Data rows         :  {n}")
    out(f"Input range  J1   : [{J1_deg.min():.3f}, {J1_deg.max():.3f}] deg")
    out(f"             J2   : [{J2_deg.min():.3f}, {J2_deg.max():.3f}] deg")
    out(f"             pos6 : [{pos6_deg.min():.3f}, {pos6_deg.max():.3f}] deg")
    out()

    out("--- Step 1A: J3 derived values ---")
    J3_vals = pos6_to_j3(J2_deg, pos6_deg)   # vectorised numpy call
    out(f"  J3 = J2 - pos6  range : [{J3_vals.min():.3f}, {J3_vals.max():.3f}] deg")
    out(f"  First 5 J3 values     : {J3_vals[:5].round(4).tolist()}")
    out()

    out("--- Step 2: FK prediction errors (mm) ---")
    out(f"  Mean absolute error (MAE) 3D : {err_3d.mean():.6e}")
    out(f"  RMS error                3D  : {np.sqrt((err_3d**2).mean()):.6e}")
    out(f"  Max absolute error       3D  : {err_3d.max():.6e}")
    out(f"  Per-axis mean |error|    x   : {np.abs(err_x).mean():.6e}")
    out(f"                           y   : {np.abs(err_y).mean():.6e}")
    out(f"                           z   : {np.abs(err_z).mean():.6e}")
    out()

    out("--- Sample predictions vs ground truth (first 5 rows) ---")
    out(f"  {'row':>3}  {'x_true':>9} {'x_pred':>9}  "
        f"{'y_true':>9} {'y_pred':>9}  "
        f"{'z_true':>9} {'z_pred':>9}  {'err_3d':>12}")
    for i in range(5):
        out(f"  {i:>3}  {x_true[i]:>9.4f} {x_pred[i]:>9.4f}  "
            f"{y_true[i]:>9.4f} {y_pred[i]:>9.4f}  "
            f"{z_true[i]:>9.4f} {z_pred[i]:>9.4f}  "
            f"{err_3d[i]:>12.6e}")
    out()

    # Worst 3 rows
    worst = np.argsort(err_3d)[-3:][::-1]
    out("--- Worst 3 rows ---")
    out(f"  {'row':>3}  {'J1':>8} {'J2':>8} {'pos6':>8}  "
        f"{'x_true':>9} {'x_pred':>9}  {'err_3d':>12}")
    for i in worst:
        out(f"  {i:>3}  {J1_deg[i]:>8.4f} {J2_deg[i]:>8.4f} {pos6_deg[i]:>8.4f}  "
            f"{x_true[i]:>9.4f} {x_pred[i]:>9.4f}  {err_3d[i]:>12.6e}")
    out()

    # Verdict
    max_e = err_3d.max()
    if max_e < 1e-3:
        out("VERDICT: PASS")
        out(f"  Maximum 3D error {max_e:.2e} mm is at floating-point precision.")
        out("  The FK equations are analytically exact for this dataset.")
        out("  Errors reflect only floating-point rounding, not model inaccuracy.")
    elif max_e < 1.0:
        out(f"VERDICT: WARN  (max error {max_e:.4f} mm)")
        out("  Sub-millimetre but above floating-point floor.")
        out("  Check L3 value or angle sign conventions.")
    else:
        out(f"VERDICT: FAIL  (max error {max_e:.4f} mm)")
        out("  Errors exceed 1 mm — review FK derivation.")
    out("=" * 65)

    return lines


if __name__ == "__main__":
    # Tee output to file AND console
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
        def flush(self):
            for st in self.streams:
                st.flush()

    f = open(OUT_FILE, 'w')
    sys.stdout = Tee(sys.__stdout__, f)

    try:
        data = load_data(DATA_FILE)
        run_validation(data)
    except FileNotFoundError as e:
        print(e)
    finally:
        sys.stdout = sys.__stdout__
        f.close()
        print(f"\nResults also saved to: {OUT_FILE}")
