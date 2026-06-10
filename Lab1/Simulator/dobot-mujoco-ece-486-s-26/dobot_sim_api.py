"""
DOBOT Simulator API Module.
Contains kinematics, coordinate mapping, and motion control logic.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from dobot_mujoco.env.dobot_cube_stack import DOBOT_MOTOR_LIMITS
from dobot_mujoco.env.dobot_pick_place import DobotPickPlace


EE_GEOM_NAME = "suctionCup_link2"
ARM_JOINT_NAMES = ["motor1", "motor2", "motor3", "motor4"]
ARM_LIMITS = np.array(DOBOT_MOTOR_LIMITS[:4], dtype=np.float64)
HOME_CTRL = np.zeros(4, dtype=np.float64)


@dataclass
class SimDobotAPI:
    env: DobotPickPlace
    viewer: object | None
    home_pos: np.ndarray
    base_pos_mm: np.ndarray
    suction_on: bool = False

    def sync_viewer(self, every: int = 1, step: int = 0) -> None:
        if self.viewer is not None and step % every == 0:
            self.viewer.sync()
            time.sleep(0.002)

    def current_xyz_m(self) -> np.ndarray:
        geom_id = self.env.model.geom(EE_GEOM_NAME).id
        return self.env.data.geom_xpos[geom_id].copy()

    def current_xyz_mm(self) -> np.ndarray:
        # Return coordinates in the User Robot Frame (X=Forward, Y=Side, Z=Up)
        # Sim X (Side) -> User Y
        # Sim Y (Forward) -> User -X
        # Sim Z (Up) -> User Z + 62.8 (Calibrated)
        sim_pos = self.current_xyz_m() * 1000.0 - self.base_pos_mm
        user_x = -sim_pos[1]
        user_y = sim_pos[0]
        user_z = sim_pos[2] - 62.8
        return np.array([user_x, user_y, user_z])

    def current_joint_deg(self) -> np.ndarray:
        joint_rad = np.array(
            [self.env.data.joint(name).qpos[0] for name in ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        return np.rad2deg(joint_rad)

    def get_pose(self) -> np.ndarray:
        xyz_mm = self.current_xyz_mm()
        joints_deg = self.current_joint_deg()
        tool_roll_deg = float(joints_deg[3])
        return np.array(
            [
                xyz_mm[0],
                xyz_mm[1],
                xyz_mm[2],
                tool_roll_deg,
                joints_deg[0],
                joints_deg[1],
                joints_deg[2],
                joints_deg[3],
            ],
            dtype=np.float64,
        )


def _step_env(api: SimDobotAPI, action: np.ndarray, max_steps: int, done_fn) -> None:
    for step in range(max_steps):
        api.env.step(action)
        api.sync_viewer(every=1, step=step)
        if done_fn():
            break


def _compute_cartesian_action(
    api: SimDobotAPI,
    desired_pos_robot_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical IK solver based on the student's robot code.
    """
    # Robot frame coordinates from user: (X=Forward, Y=Side, Z=Up)
    user_x, user_y, user_z = desired_pos_robot_mm
    
    # Constants from user's code
    L1 = 135.0
    L2 = 147.0
    toolOffset = 59.7
    
    # Calibrated Lbase for simulation: 
    # User Z = 0 corresponds to Sim Robot Frame Z = 62.8
    z_for_kinematics = user_z 

    try:
        # theta1
        theta1 = math.atan2(user_y, user_x)
        
        # Wrist position
        xWrist = user_x - toolOffset * math.cos(theta1)
        yWrist = user_y - toolOffset * math.sin(theta1)
        
        # Triangle calculations
        L_tri = math.sqrt(xWrist**2 + yWrist**2 + z_for_kinematics**2)
        
        def safe_acos(val):
            return math.acos(max(-1.0, min(1.0, val)))

        phi2 = safe_acos(-(L2**2 - L_tri**2 - L1**2) / (2 * L_tri * L1))
        psi = math.asin(max(-1.0, min(1.0, z_for_kinematics / L_tri)))

        # theta2 and theta3
        theta2_rad = math.pi/2 - phi2 - psi
        tst = -((z_for_kinematics - L1 * math.cos(theta2_rad)) / L2)
        theta3_rad = math.asin(max(-1.0, min(1.0, tst)))
        
        target_q = np.array([theta1, theta2_rad, theta3_rad, 0.0])
        current_q = np.array([api.env.data.joint(name).qpos[0] for name in ARM_JOINT_NAMES[:4]])
        
        # PD control in joint space towards the IK target
        q_err = target_q - current_q
        dq = q_err * 0.5 # Gain
        
        action = np.zeros(5, dtype=np.float32)
        action[:4] = np.clip(dq / ARM_LIMITS, -1.0, 1.0)
        action[4] = 1.0 if api.suction_on else -1.0
        
        # Map back to Sim Robot Frame for internal error reporting if needed
        sim_pos = np.array([user_y, -user_x, user_z + 62.8])
        
        return action, sim_pos, q_err
        
    except Exception:
        return np.zeros(5, dtype=np.float32), api.current_xyz_mm(), np.zeros(3)


def move_to_xyz(api: SimDobotAPI, x: float, y: float, z: float) -> None:
    """Move the robot to a Cartesian target in the Robot Frame (mm)."""
    target_robot_mm = np.array([x, y, z], dtype=np.float64)

    def done() -> bool:
        err_mm = np.linalg.norm(api.current_xyz_mm() - target_robot_mm)
        return err_mm < 3.0

    for step in range(1200):
        action, _, _ = _compute_cartesian_action(api, target_robot_mm)
        api.env.step(action)
        api.sync_viewer(every=1, step=step)
        if done():
            break


def move_joint_angles(api: SimDobotAPI, J1: float, J2: float, J3: float, J4: float = 0.0) -> None:
    """Move the robot to specific joint angles (degrees)."""
    target_rad = np.deg2rad(np.array([J1, J2, J3, J4], dtype=np.float64))

    def done() -> bool:
        current_rad = np.array(
            [api.env.data.joint(name).qpos[0] for name in ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        err_deg = np.max(np.abs(np.rad2deg(current_rad - target_rad)))
        return err_deg < 1.0

    for step in range(1000):
        api.env.suction_activated = api.suction_on
        api.env.data.ctrl[:4] = target_rad
        api.env.data.ctrl[4] = 1.0 if api.suction_on else 0.0
        mj.mj_step(api.env.model, api.env.data)
        api.sync_viewer(every=1, step=step)
        if done():
            break


def move_to_home(api: SimDobotAPI) -> None:
    """Return the robot to its home joint configuration."""
    move_joint_angles(api, 0.0, 0.0, 0.0, 0.0)


def rotate_end_effector(api: SimDobotAPI, angle: float) -> None:
    """Rotate the end effector (J4) to a specific angle (degrees)."""
    if -90.0 <= angle <= 90.0:
        j1, j2, j3, _ = api.current_joint_deg()
        move_joint_angles(api, j1, j2, j3, angle)


def engage_suction(api: SimDobotAPI) -> None:
    """Activate the suction cup."""
    api.suction_on = True
    hold = np.zeros(5, dtype=np.float32)
    hold[4] = 1.0
    _step_env(api, hold, max_steps=60, done_fn=lambda: False)


def release_suction(api: SimDobotAPI) -> None:
    """Deactivate the suction cup."""
    api.suction_on = False
    hold = np.zeros(5, dtype=np.float32)
    hold[4] = -1.0
    _step_env(api, hold, max_steps=60, done_fn=lambda: False)


def stop_pump(api: SimDobotAPI) -> None:
    """Stop the suction pump."""
    release_suction(api)


def get_pose(api: SimDobotAPI) -> np.ndarray:
    """Get the current pose [X, Y, Z, R, J1, J2, J3, J4]."""
    return api.get_pose()
