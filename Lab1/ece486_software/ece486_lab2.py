"""
ece486_lab2.py  -  Lab 2 real-robot script  (ECE 486)
======================================================
Builds directly on top of ece486_starter_code.py functions.
Functions are redeclared here (rather than imported) because
ece486_starter_code.py runs Lab 1 trajectory code at module level,
which would execute on import.

What this file does
  Step 5  : joint_workspace_check(J1, J2, J3) — check a joint configuration
             against the Lab 1 workspace via FK.
  Step 6  : run_lab2_validation(api) — move the real robot through 8 test
             configurations, read back the pose, compare FK prediction vs
             robot-reported (x, y, z), and save results to a JSON file.

HOW TO RUN:
    On the lab PC, place this file AND fk.py in the same folder as the
    DLL files (same folder as ece486_starter_code.py).
    Then run:   python ece486_lab2.py
"""

import DobotDllType as dType
import numpy as np
import json
import sys
from fk import forward_kinematics, pos6_to_j3, j3_to_pos6

# ── Robot setup (identical to ece486_starter_code.py) ─────────────────────────
api      = dType.load()
home_pos = [200, 100, 50]


# ── Lab 1 functions (redeclared to avoid importing module-level code) ──────────

def initialize_robot(api):
    com_port = dType.SearchDobot(api)
    print(dType.SearchDobot(api))
    if "COM" not in com_port[0]:
        print("Error: robot not found. Exiting.")
        exit()
    state = dType.DobotConnect.DobotConnect_NoError
    for i in range(len(com_port)):
        state_full = dType.ConnectDobot(api, com_port[i], 115200)
        state = state_full[0]
        if state == dType.DobotConnect.DobotConnect_NoError:
            print("Connected!")
            break
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("Cannot connect. Exiting.")
        exit()
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    dType.SetPTPCommonParams(api, 50, 50, isQueued=1)
    dType.SetHOMEParams(api, home_pos[0], home_pos[1], home_pos[2], 0, isQueued=1)
    execCmd = dType.SetHOMECmd(api, temp=0, isQueued=1)[0]
    dType.SetQueuedCmdStartExec(api)
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)
    print("Robot ready.")


def move_to_xyz(api, x, y, z):
    execCmd = dType.SetPTPCmd(
        api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, 0, isQueued=0)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)


def move_joint_angles(api, J1, J2, forearm_angle, J4=0):
    """
    Move the robot to the given joint angles.
    NOTE: the third argument is the FOREARM-GROUND ANGLE (pos6), not J3.
    To move to a specific J3, convert first: pos6 = j3_to_pos6(J2, J3).
    """
    execCmd = dType.SetPTPCmd(
        api, dType.PTPMode.PTPMOVJANGLEMode,
        J1, J2, forearm_angle, J4, isQueued=0)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)


def move_to_home(api):
    move_to_xyz(api, home_pos[0], home_pos[1], home_pos[2])


def is_point_safe(x, y, z):
    """Lab 1 workspace check (Cartesian). Returns True if (x,y,z) is safe."""
    radius = np.sqrt(x**2 + y**2)
    return (-120 <= z <= 0) and (140 <= radius <= 260) and (x >= 0)


def is_path_safe(current_pose, target_pose, steps=10):
    """Lab 1 path safety check via linear interpolation."""
    x1, y1, z1 = current_pose[0], current_pose[1], current_pose[2]
    x2, y2, z2 = target_pose[0], target_pose[1], target_pose[2]
    for i in range(steps):
        t = i / (steps - 1)
        if not is_point_safe(x1 + t*(x2-x1), y1 + t*(y2-y1), z1 + t*(z2-z1)):
            return False
    return True


# ── Lab 2 additions ────────────────────────────────────────────────────────────

def joint_workspace_check(J1_deg, J2_deg, J3_deg):
    """
    Step 5: Check whether joint configuration (J1, J2, J3) places the
    end-effector inside the restricted workspace.

    Strategy: convert J3 → pos6, apply FK to get (x, y, z), then run the
    existing Cartesian is_point_safe check. This reuses the Lab 1 workspace
    geometry without re-deriving limits in joint space (which is hard).

    Args:
        J1_deg, J2_deg : joint angles in degrees
        J3_deg         : TRUE J3 joint angle (NOT pos6 / forearm-ground angle)
    Returns:
        (safe, x, y, z) where safe is bool and x/y/z is the FK-predicted position
    """
    pos6 = j3_to_pos6(J2_deg, J3_deg)
    x, y, z = forward_kinematics(J1_deg, J2_deg, pos6)
    safe = is_point_safe(x, y, z)
    return safe, x, y, z


def move_joint_safe(api, J1, J2, J3):
    """
    Check joint configuration, then move the robot if safe.
    Prints a decision line for every call.

    Args:
        J1, J2 : joint angles in degrees
        J3     : TRUE J3 joint angle in degrees (converted to pos6 internally)
    Returns:
        True if the robot was moved, False if the request was rejected.
    """
    safe, x_pred, y_pred, z_pred = joint_workspace_check(J1, J2, J3)
    pos6 = j3_to_pos6(J2, J3)

    if not safe:
        r = np.sqrt(x_pred**2 + y_pred**2)
        print(f"[joint_ws] REJECTED  J1={J1} J2={J2} J3={J3} pos6={pos6:.2f}"
              f"  FK→ ({x_pred:.2f}, {y_pred:.2f}, {z_pred:.2f})  r={r:.2f}")
        return False

    print(f"[joint_ws] OK        J1={J1} J2={J2} J3={J3} pos6={pos6:.2f}"
          f"  FK→ ({x_pred:.2f}, {y_pred:.2f}, {z_pred:.2f})")
    move_joint_angles(api, J1, J2, pos6)
    return True


# ── Step 6 test configurations ─────────────────────────────────────────────────
# Each entry: (label, J1_deg, J2_deg, J3_deg)
# All 8 are pre-verified to be within the workspace.
# They cover: interior, arm raised, rotated left/right, deep z, short reach,
# very deep z, and combined J1/J2 variation.
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


def run_lab2_validation(api, out_file="lab2_validation_results.json"):
    """
    Step 6: Drive the robot through each test configuration and compare the
    FK-predicted (x, y, z) against what GetPose reports.

    For each config:
      1. Verify the joint config is safe (joint_workspace_check).
      2. Move the robot using move_joint_angles (passing pos6, not J3).
      3. Read back the pose: GetPose() → [x, y, z, r, J1, J2, pos6, J4].
      4. Apply our FK to the REPORTED joint angles (pose[4], pose[5], pose[6]).
      5. Compute error between our FK prediction and the robot-reported (x, y, z).
    """
    results = []
    errors_3d = []

    print("\n" + "=" * 70)
    print("LAB 2  STEP 6  --  FK VALIDATION ON REAL ROBOT")
    print("=" * 70)

    for label, J1, J2, J3 in VALIDATION_CONFIGS:
        print(f"\n--- {label} ---")
        safe, x_fk_pre, y_fk_pre, z_fk_pre = joint_workspace_check(J1, J2, J3)
        if not safe:
            print(f"  SKIPPED (joint_workspace_check rejected this config)")
            continue

        pos6_cmd = j3_to_pos6(J2, J3)
        print(f"  Commanding: J1={J1}  J2={J2}  pos6={pos6_cmd:.3f}  (J3={J3})")
        print(f"  FK predicts: ({x_fk_pre:.3f}, {y_fk_pre:.3f}, {z_fk_pre:.3f}) mm")

        move_joint_angles(api, J1, J2, pos6_cmd)
        pose = list(dType.GetPose(api))   # [x, y, z, r, J1, J2, pos6, J4]

        act_x, act_y, act_z = pose[0], pose[1], pose[2]
        rep_J1, rep_J2, rep_pos6 = pose[4], pose[5], pose[6]

        # Apply OUR FK to the robot's reported joint angles
        x_fk_rep, y_fk_rep, z_fk_rep = forward_kinematics(
            rep_J1, rep_J2, rep_pos6)

        # Error: our FK applied to reported joints vs robot-reported Cartesian
        err = np.sqrt((x_fk_rep - act_x)**2 +
                      (y_fk_rep - act_y)**2 +
                      (z_fk_rep - act_z)**2)
        errors_3d.append(err)

        print(f"  Robot reports: ({act_x:.3f}, {act_y:.3f}, {act_z:.3f}) mm")
        print(f"  Our FK on reported joints: ({x_fk_rep:.3f}, {y_fk_rep:.3f}, "
              f"{z_fk_rep:.3f}) mm")
        print(f"  3D error: {err:.6e} mm")

        results.append({
            "label":          label,
            "commanded_J1":   J1,
            "commanded_J2":   J2,
            "commanded_J3":   J3,
            "commanded_pos6": pos6_cmd,
            "fk_predicted":   [x_fk_pre, y_fk_pre, z_fk_pre],
            "reported_pose":  pose,
            "reported_J1":    rep_J1,
            "reported_J2":    rep_J2,
            "reported_pos6":  rep_pos6,
            "fk_on_reported": [x_fk_rep, y_fk_rep, z_fk_rep],
            "error_3d_mm":    err,
        })

        move_to_home(api)   # return home between tests

    # Summary
    if errors_3d:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Tests run          : {len(errors_3d)}")
        print(f"  Mean 3D error      : {np.mean(errors_3d):.6e} mm")
        print(f"  Max  3D error      : {np.max(errors_3d):.6e} mm")
        print(f"  Dobot spec (repeat): 0.2 mm")
        if np.max(errors_3d) < 0.2:
            print("  RESULT: meets or exceeds the 0.2 mm repeatability spec.")

    # Save to JSON
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_file}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize_robot(api)
    run_lab2_validation(api)
