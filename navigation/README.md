# ShepherdRover Navigation Stack

This directory contains the ROS2-based navigation system for ShepherdRover.

## Overview

The navigation stack provides autonomous navigation capabilities for agricultural field scouting, including:

- **SLAM (Simultaneous Localization and Mapping)** for field mapping
- **Path Planning** for efficient field coverage
- **Obstacle Avoidance** for safe operation
- **GPS Integration** for absolute positioning

## Package Structure

```
navigation/
├── shepherd_navigation/          # Main navigation package
├── shepherd_slam/               # SLAM and mapping
├── shepherd_planning/           # Path planning algorithms
├── shepherd_control/            # Motor control and kinematics
├── shepherd_sensors/            # Sensor drivers and fusion
└── shepherd_launch/             # Launch files and configurations
```

## Quick Start

### Prerequisites

- ROS2 Humble or newer
- Python 3.8+
- Required dependencies (see `package.xml`)

### Build

```bash
cd navigation
colcon build
source install/setup.bash
```

### Run

```bash
# Basic navigation stack
ros2 launch shepherd_navigation bringup.launch.py

# With SLAM for mapping
ros2 launch shepherd_slam mapping.launch.py

# With path planning for field coverage
ros2 launch shepherd_planning field_coverage.launch.py
```

## Configuration

- `config/` - Navigation parameters and configurations
- `maps/` - Pre-built field maps (if available)
- `launch/` - Launch file configurations

## Topics

### Subscribed Topics
- `/cmd_vel` - Velocity commands
- `/scan` - Laser scan data
- `/imu/data` - IMU data
- `/gps/fix` - GPS position

### Published Topics
- `/map` - Occupancy grid map
- `/path` - Planned path
- `/robot_pose` - Current robot pose
- `/diagnostics` - System diagnostics

## Development

See `CONTRIBUTING.md` for development guidelines and testing procedures. 