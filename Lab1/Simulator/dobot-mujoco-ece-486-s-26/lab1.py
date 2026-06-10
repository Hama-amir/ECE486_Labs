import numpy as np
from sim_starter import create_sim_api, initialize_robot
from dobot_sim_api import move_to_xyz, get_pose
import json

# Initialization 
api = create_sim_api(seed=0, headless=False)
initialize_robot(api)

def is_point_safe(x, y, z):
    """Returns True if the (x,y,z) coordinate is within the safe semi-annular prism."""
    radius = np.sqrt(x**2 + y**2)
    
    # Check Z boundaries (-120mm to 0mm)
    z_safe = (-120 <= z <= 0)
    
    # Check Radius boundaries (140mm to 260mm)
    radius_safe = (140 <= radius <= 260)
    
    # Check X boundary (x >= 0)
    x_safe = (x >= 0)
    
    # Return boolean result
    return z_safe and radius_safe and x_safe

def is_path_safe(current_pose, target_pose, steps=10):
    """Interpolates between current and target poses to ensure the whole line is safe."""
    # Extract x, y, z from current_pose
    x1, y1, z1 = current_pose[0], current_pose[1], current_pose[2]
    x2, y2, z2 = target_pose[0], target_pose[1], target_pose[2]

    # Generate an array of 'steps' points between current and target
    x_points = np.linspace(x1, x2, steps)
    y_points = np.linspace(y1, y2, steps)
    z_points = np.linspace(z1, z2, steps)

    # Loop through points; if any fail is_point_safe(), return False
    for i in range(steps):
        if not is_point_safe(x_points[i], y_points[i], z_points[i]):
            return False

    # If all pass, return True
    return True

def generate_circle_trajectory(radius=40, num_points=50):
    """Generates a list of (x,y,z) targets forming a circle."""
    trajectory = []
    t_values = np.linspace(0, 2*np.pi, num_points)
    
    for t in t_values:
        # Calculate x, y, z using your parametric equations
        x = 200 + radius * np.cos(t)
        y = radius * np.sin(t)
        z = -100
        trajectory.append((x, y, z))
    
    return trajectory
        

trajectory_targets = generate_circle_trajectory()
all_runs_data = []
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
print("Moving from Home to start of trajectory...")
start_target = trajectory_targets[0]
move_to_xyz(api, start_target[0], start_target[1], start_target[2])

# 10 full runs of trajectory
for run in range(10):
    run_data = []
    print("Starting run", run + 1)

    for i, target in enumerate(trajectory_targets):
        current_pose = get_pose(api) 
        
        if is_path_safe(current_pose, target):
            move_to_xyz(api, target[0], target[1], target[2])
            actual_pose = get_pose(api)
            serializable_pose = [float(val) for val in actual_pose]
            run_data.append(serializable_pose)

            # Print a progress update every 10 points so you know it isn't frozen!
            if (i + 1) % 10 == 0:
                print(f"  Reached point {i + 1}/50...")
        else:
            print(f"Path to {target} is unsafe! Stopping.")
            break
            
    all_runs_data.append(run_data)

# Save all_runs_data to a JSON file
print("Saving data to JSON...")
with open('trajectory_runs_data.json', 'w') as f:
    json.dump(all_runs_data, f, indent=4)
print("Simulation Complete!")