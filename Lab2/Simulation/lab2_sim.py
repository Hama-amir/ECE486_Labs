"""
lab2_sim.py  -  Lab 2 simulator script  (ECE 486)  [Gate 2 deliverable]
=======================================================================
Demonstrates joint-space motion in the MuJoCo simulator and validates the
FK model, satisfying the two pre-lab requirements before bench access:

  Gate 2a: show joint-space motion working in the sim.
  Gate 2b: show the FK pipeline + a stated validation plan.

HOW TO RUN (from the repo root):
    uv run lab2_sim.py
    uv run lab2_sim.py --headless

Requires fk.py and workspace_function.py in the same folder.
"""

import argparse
import numpy as np

from sim_starter    import create_sim_api, initialize_robot
from dobot_sim_api  import (move_joint_angles, get_pose,
                             move_to_home, stop_pump)
from workspace_function import in_workspace
from fk import forward_kinematics, j3_to_pos6, pos6_to_j3


# ── Workspace check (reuses Lab 1 function via FK) ────────────────────────────

def joint_workspace_check(J1_deg, J2_deg, J3_deg):
    """
    Step 5: Check whether (J1, J2, J3) is inside the restricted workspace.
    Converts J3 → pos6, applies FK to get (x, y, z), then calls in_workspace.

    Returns: (safe: bool, x: float, y: float, z: float)
    """
    pos6 = j3_to_pos6(J2_deg, J3_deg)
    x, y, z = forward_kinematics(J1_deg, J2_deg, pos6)
    safe = in_workspace(x, y, z)
    return safe, x, y, z


# ── Test configurations (same 8 as the real-robot script) ─────────────────────
VALIDATION_CONFIGS = [
    ("Interior / forward",    0,  60,  -5),
    ("Arm raised, low reach", 0,  40, -25),
    ("Rotated left  J1=+8",   8,  60,  -5),
    ("Rotated right J1=-8",  -8,  60,  -5),
    ("Low arm, deep z",       0,  70,   3),
    ("Short reach",           0,  50, -25),
    ("Very deep z",           0,  75,   5),
    ("J1 + J2 combined",      5,  50, -15),
]


# ── Validation loop ───────────────────────────────────────────────────────────

def run_sim_validation(api):
    """
    Gate 2 demo: drive the sim through every test configuration, read back
    the pose, apply our FK to the reported joint angles, and compare with
    the robot's reported (x, y, z).
    """
    print("\n" + "=" * 70)
    print("LAB 2 SIM  --  Joint-space motion + FK validation")
    print("=" * 70)

    errors = []

    for label, J1, J2, J3 in VALIDATION_CONFIGS:
        safe, x_pred, y_pred, z_pred = joint_workspace_check(J1, J2, J3)
        pos6_cmd = j3_to_pos6(J2, J3)

        print(f"\n  [{label}]")
        print(f"    J1={J1}  J2={J2}  J3={J3}  pos6={pos6_cmd:.2f}"
              f"  FK→ ({x_pred:.2f}, {y_pred:.2f}, {z_pred:.2f})  "
              f"safe={safe}")

        if not safe:
            print("    SKIPPED (outside workspace)")
            continue

        # Joint-space move in the sim
        move_joint_angles(api, J1, J2, pos6_cmd, 0.0)

        # Read back pose: [x, y, z, r, J1, J2, pos6, J4]
        pose = get_pose(api)
        act_x, act_y, act_z = float(pose[0]), float(pose[1]), float(pose[2])
        rep_J1  = float(pose[4])
        rep_J2  = float(pose[5])
        rep_pos6= float(pose[6])

        # Our FK on the reported joint angles
        x_fk, y_fk, z_fk = forward_kinematics(rep_J1, rep_J2, rep_pos6)
        err = np.sqrt((x_fk - act_x)**2 + (y_fk - act_y)**2 + (z_fk - act_z)**2)
        errors.append(err)

        # Also derive J3 from what the robot reported (Step 1A demo)
        rep_J3 = pos6_to_j3(rep_J2, rep_pos6)

        print(f"    Robot reported:     ({act_x:.3f}, {act_y:.3f}, {act_z:.3f}) mm")
        print(f"    Reported joints:    J1={rep_J1:.3f}  J2={rep_J2:.3f}  "
              f"pos6={rep_pos6:.3f}  (J3={rep_J3:.3f})")
        print(f"    Our FK on reported: ({x_fk:.3f}, {y_fk:.3f}, {z_fk:.3f}) mm")
        print(f"    3D error:           {err:.6e} mm")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Configs tested     : {len(errors)}")
    if errors:
        print(f"  Mean 3D error      : {np.mean(errors):.6e} mm  (expect ~float eps in sim)")
        print(f"  Max  3D error      : {np.max(errors):.6e} mm")
    print()
    print("Validation plan for real robot:")
    print("  - Use the same 8 configurations on the real Dobot.")
    print("  - Verify joint_workspace_check accepts each one.")
    print("  - Command via move_joint_angles(api, J1, J2, pos6).")
    print("  - Compare our FK result with GetPose() Cartesian output.")
    print("  - Configurations were chosen to cover:")
    print("      * Mid-range interior (configs 1, 8)")
    print("      * Arm elevated, lower reach (config 2)")
    print("      * J1 rotation left and right (configs 3, 4)")
    print("      * Deep z (configs 5, 7)")
    print("      * Short reach near r=200mm (config 6)")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lab 2 sim — FK + joint-space demo")
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    api = create_sim_api(seed=args.seed, headless=args.headless)
    try:
        initialize_robot(api)
        run_sim_validation(api)
        move_to_home(api)
    finally:
        stop_pump(api)
        if api.viewer is not None:
            api.viewer.close()
        api.env.close()


if __name__ == "__main__":
    main()
