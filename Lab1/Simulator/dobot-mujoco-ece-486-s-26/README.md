# Dobot Magician MuJoCo Simulation Environment

This repository provides a (reasonably...) high-fidelity MuJoCo simulation of the Dobot Magician, calibrated to match the kinematics and coordinate system of the physical robots used in the lab. It has been tested on Linux and Windows. There is no reason it shouldn't work on Mac, but we can work together to fix any issues that arise.

## 1. Prerequisites

Before starting, ensure you have the following installed:
*   **Git**: [Install Git](https://git-scm.com/downloads)
*   **uv**: The extremely fast Python package manager.
    *   **Windows (PowerShell):** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    *   **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Note:** if you don't like using uv, you can do it all via pure Python, just install the requirements and run it. However, uv is what we have tested this on.

## 2. Installation

### Step 1: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://git.uwaterloo.ca/mstachow/dobot-mujoco-ece-486-s-26
cd dobot-mujoco-ece-486-s-26
```

### Step 2: Sync the Environment
This command will automatically create a virtual environment and install all dependencies (MuJoCo, NumPy, Gymnasium) using the `uv` tool:
```bash
uv sync
```

## 3. Running the Simulator

To verify your installation, run the starter script. This will open the MuJoCo viewer and demonstrate both Cartesian and Joint-space movements.

```bash
uv run sim_starter.py
```

## 4. Understanding the Simulation

The simulation has been calibrated so that the code you write here can be ported directly to the real robot. The axes are set up identically to the real robot, and the simulation has been tested to ensure that its kinematics match data from the real robot. Therefore, you should attemp most of your labs on this simulator before coming to the robots, since robot time is limited! All of the functions you will use to control the real robot exist in the simulator with the same names, and the process to intialize and move the robot is the same. You should need only minimal changes to your code (for example, using the correct get pose function) when you work with the real robot. 

That being said, there are some...interesting quirks to the simulator that you should know:

1. The suction cup has magical telekenetic powers. As long as it is vaguely touching the blocks, it can pick them up. Don't rely on the simulator as it is currently written and assume that you can just port actual object interactions to the real robot.
2. For some reason, the simulation will segfault sometimes when it is done. We're actively working on a fix for this, but the segfault only happens once the simulation is done, so it shouldn't affect your results.
3. The simulator moves *much* faster than the robot is allowed to. This is just so you aren't sitting there forever waiting for the simulated robot to move. When you're in the real lab, remember that the robot will be slower.


## 5. Development Workflow

When you are working on a new lab, you should probably start from `sim_starter.py`. We recommend not modifying that file directly, since if you break something it will just take a few more steps to get it back from gitlab.

1.  **Copy the Template:** Use `sim_starter.py` as a reference for your own scripts.
2.  **Import the API:** Always include `from dobot_sim_api import ...` at the top of your scripts.
3.  **Initialize:** Use the `create_sim_api()` and `initialize_robot()` functions to set up the environment.
4.  **Control:** Use `move_to_xyz()` for coordinate control or `move_joint_angles()` for direct motor control.

### Key API Commands:
*   `get_pose(api)`: Returns `[X, Y, Z, R, J1, J2, J3, J4]`
*   `move_to_xyz(api, x, y, z)`: Moves end-effector to robot-frame coordinates, in mm.
*   `move_joint_angles(api,J1,J2,J3,J4)`: Moves end-effector by directly setting the joint angles in degrees.
*   `engage_suction(api)` / `release_suction(api)`: Controls the suction cup.
