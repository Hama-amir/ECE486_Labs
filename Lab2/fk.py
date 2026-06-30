"""
fk.py  -  Forward Kinematics module for the Dobot Magician  (ECE 486 Lab 2)
===========================================================================
Standalone module with no robot or simulation dependencies.
Import this file in every Lab 2 script.

Dobot Magician link lengths
  L1 = 135 mm  (upper arm, base joint to elbow)
  L2 = 147 mm  (forearm, elbow to wrist)
  L3 =  59.7mm (end-effector offset, wrist to tool-centre)

Angle conventions (all angles in DEGREES, converted internally to radians)
  J1  : rotation of the whole arm about the vertical z-axis.
          J1 = 0 means the arm faces the positive x-direction.
  J2  : elevation angle of the upper arm.
          J2 = 0 means the upper arm points straight up (perpendicular to ground).
          Increasing J2 tips the arm forward toward the table.
  J3  : the TRUE physical elbow joint angle, derived from what the robot reports.
          J3 = 0 when the upper arm is vertical (J2 = 0) and the forearm is
          horizontal (pos[6] = 0).  CCW rotation increases J3.
  pos6: the forearm-ground angle that the Dobot API stores in GetPose()[6].
          pos6 > 0 means the forearm is angled BELOW the horizontal.
          This is what the robot uses internally and what move_joint_angles takes.

Step 1A  J3   = J2 - pos6   (derive joint angle from what the robot reports)
Step 1B  pos6 = J2 - J3     (derive robot command from desired joint angle)
"""

import math

# ── Link lengths (mm) ──────────────────────────────────────────────────────────
L1 = 135.0   # upper arm
L2 = 147.0   # forearm
L3 =  59.7   # end-effector offset


# ── Step 1A ────────────────────────────────────────────────────────────────────
def pos6_to_j3(J2_deg, pos6_deg):
    """
    Derive the true J3 joint angle from the forearm-ground angle reported by
    the robot (GetPose()[6]) and J2.

    Relationship:  J3 = J2 - pos6

    Verified reference condition:
      - J2 = 0  (upper arm vertical)  AND  pos6 = 0  (forearm horizontal)
        => J3 = 0 ✓
      - Increasing J3 (CCW) rises the forearm (pos6 decreases) ✓

    Args:
        J2_deg   : J2 angle from GetPose()[5], degrees
        pos6_deg : forearm-ground angle from GetPose()[6], degrees
    Returns:
        J3_deg   : true elbow joint angle, degrees
    """
    return J2_deg - pos6_deg


# ── Step 1B ────────────────────────────────────────────────────────────────────
def j3_to_pos6(J2_deg, J3_deg):
    """
    Convert a desired true J3 joint angle to the forearm-ground angle (pos6)
    that move_joint_angles() expects.

    Relationship:  pos6 = J2 - J3   (inverse of Step 1A)

    Args:
        J2_deg  : J2 angle, degrees
        J3_deg  : desired true J3 angle, degrees
    Returns:
        pos6_deg: forearm-ground angle to pass to move_joint_angles(), degrees
    """
    return J2_deg - J3_deg


# ── Step 2 ─────────────────────────────────────────────────────────────────────
def forward_kinematics(J1_deg, J2_deg, pos6_deg):
    """
    Compute the 3D end-effector position (x, y, z) in mm from joint angles.

    Derivation:
      Working first in the vertical (r, z) plane for a fixed J1:

        r = L1·sin(J2) + L2·cos(pos6) + L3        [horizontal reach, mm]
        z = L1·cos(J2) − L2·sin(pos6)              [height, mm]

      J1 then rotates that planar reach into 3D:

        x = r · cos(J1)
        y = r · sin(J1)

    Intuition:
      - L1·sin(J2) : how far the ELBOW is from the base axis horizontally.
        (J2=0 → elbow directly above base, no horizontal reach from link 1.)
      - L2·cos(pos6): how far the WRIST is from the elbow horizontally.
        (pos6=0 → forearm horizontal → full L2 contributes to reach.)
      - L3          : fixed end-effector offset along the tool axis.
      - The z terms mirror the pattern but with cos/sin swapped, and the
        forearm's z contribution is negative because pos6>0 means the forearm
        points downward.

    Args:
        J1_deg   : J1 angle (base rotation), degrees
        J2_deg   : J2 angle (upper arm elevation), degrees
        pos6_deg : forearm-ground angle from GetPose()[6], degrees
    Returns:
        (x, y, z): end-effector position in mm, as a tuple of floats
    """
    J1   = math.radians(J1_deg)
    J2   = math.radians(J2_deg)
    pos6 = math.radians(pos6_deg)

    r = L1 * math.sin(J2) + L2 * math.cos(pos6) + L3
    x = r  * math.cos(J1)
    y = r  * math.sin(J1)
    z = L1 * math.cos(J2) - L2 * math.sin(pos6)

    return (x, y, z)


# ── Self-test (run: python fk.py) ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== fk.py self-test ===")

    # Values from Lab2DesignData.txt row 1:
    # J1=0, J2=56.913, pos6=30.089 → x≈300.000, y≈0, z≈0
    x, y, z = forward_kinematics(0.0, 56.913, 30.089)
    print(f"FK(0, 56.913, 30.089)  expect (~300, 0, 0):  "
          f"x={x:.4f}  y={y:.4f}  z={z:.6f}")
    assert abs(x - 300.0) < 0.01, "x mismatch"
    assert abs(z)         < 0.01, "z mismatch"

    # Step 1A
    J3 = pos6_to_j3(56.913, 30.089)
    print(f"pos6_to_j3(56.913, 30.089)  expect 26.824:  J3={J3:.4f}")
    assert abs(J3 - 26.824) < 0.001, "J3 mismatch"

    # Step 1B (round-trip)
    recovered = j3_to_pos6(56.913, J3)
    print(f"j3_to_pos6(56.913, {J3:.4f})  expect 30.089:  pos6={recovered:.4f}")
    assert abs(recovered - 30.089) < 0.001, "round-trip mismatch"

    print("All assertions passed.")
