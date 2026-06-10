"""
MuJoCo Simulation Starter Code for DOBOT.
Use this script to run your DOBOT experiments in the simulator.
"""

import argparse
import mujoco as mj
import numpy as np
from dobot_sim_api import (
    SimDobotAPI,
    move_to_xyz,
    move_joint_angles,
    get_pose,
    engage_suction,
    release_suction,
    stop_pump,
    move_to_home,
    HOME_CTRL
)
from dobot_mujoco.env.dobot_pick_place import DobotPickPlace

# Pick and Place joint targets (measured from demo)
PICK_JOINTS = np.array([-24.3, 54.9, 39.9, 73.7], dtype=np.float64)
PLACE_LIFT_JOINTS = np.array([33.2, 30.9, 25.2, 40.1], dtype=np.float64)
PLACE_JOINTS = np.array([39.4, 43.2, 47.7, 38.9], dtype=np.float64)


def create_sim_api(seed: int, headless: bool) -> SimDobotAPI:
    """Factory to create the simulator environment and API object."""
    env = DobotPickPlace(render_mode=None, position_jitter=0.0)
    env.reset(seed=seed)

    viewer = None
    if not headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    dobot_body_id = env.model.body("dobot").id
    base_pos_mm = env.data.xpos[dobot_body_id].copy() * 1000.0

    api = SimDobotAPI(
        env=env,
        viewer=viewer,
        home_pos=np.zeros(3, dtype=np.float64),
        base_pos_mm=base_pos_mm,
    )
    api.home_pos = api.current_xyz_mm()
    return api


def initialize_robot(api: SimDobotAPI) -> None:
    """Initialize robot state and drive to home position."""
    api.suction_on = False
    api.env.suction_activated = False
    api.env.data.ctrl[:4] = HOME_CTRL
    api.env.data.ctrl[4] = 0.0
    for step in range(250):
        mj.mj_step(api.env.model, api.env.data)
        api.sync_viewer(every=1, step=step)
    api.home_pos = api.current_xyz_mm()
    print(f"Simulator ready. home_pos = {api.home_pos.round(1).tolist()} mm")


def print_status(api: SimDobotAPI, label: str) -> None:
    """Print current robot and task status."""
    obs = api.env._get_obs()
    info = api.env._get_info(obs)
    cube_pos = api.env.data.body("pick_cube").xpos.copy() * 1000.0
    pose = get_pose(api)
    print(
        f"{label}: xyz_mm={pose[:3].round(1)} joints_deg={pose[4:].round(1)} "
        f"suction={api.suction_on} grasped={info['grasped']} success={info['is_success']} "
        f"cube_to_goal={info['cube_to_goal_distance']:.4f} cube_mm={cube_pos.round(1)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DOBOT Simulation Starter Script.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    api = create_sim_api(seed=args.seed, headless=args.headless)

    try:
        initialize_robot(api)
        print_status(api, "home")

        # Example 1: Move using Cartesian Coordinates (Analytical IK)
        print("\n--- Cartesian Motion Demo ---")
        move_to_xyz(api, 300, 0, 0)
        print_status(api, "at_target_1")
        
        move_to_xyz(api, 211, -31, -31)
        print_status(api, "at_target_2")

        # Example 2: Pick and Place sequence using Joint Angles
        print("\n--- Pick and Place Demo ---")
        move_to_home(api)
        
        move_joint_angles(api, *PICK_JOINTS)
        engage_suction(api)
        print_status(api, "grasped_cube")

        move_joint_angles(api, *PLACE_LIFT_JOINTS)
        move_joint_angles(api, *PLACE_JOINTS)
        release_suction(api)
        print_status(api, "released_cube")

        move_to_home(api)
        print_status(api, "back_home")
        
    finally:
        stop_pump(api)
        if api.viewer is not None:
            api.viewer.close()
        api.env.close()


if __name__ == "__main__":
    main()
