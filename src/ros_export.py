"""
ros_export.py — ROS 2 launch file export for ScenarioGen.

Converts a ScenarioConfig Pydantic model into a ROS 2 Python launch file
(scenario_launch.py) and supporting configs, following the same pattern as
carla_export.py.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .schema import Actor, Environment, ScenarioConfig

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

ROAD_TYPE_TO_WORLD: Dict[str, str] = {
    "intersection": "intersection.world",
    "highway": "highway.world",
    "residential": "residential.world",
    "parking_lot": "parking_lot.world",
}

ACTOR_TYPE_TO_MODEL: Dict[str, str] = {
    "vehicle": "vehicle_box",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "motorcycle": "motorcycle",
}

# weather → (precipitation_rate, fog_density, wind_speed)
WEATHER_TO_GAZEBO_PARAMS: Dict[str, Dict[str, float]] = {
    "clear": {"precipitation_rate": 0.0, "fog_density": 0.0, "wind_speed": 0.0},
    "rain":  {"precipitation_rate": 5.0, "fog_density": 0.1, "wind_speed": 2.0},
    "fog":   {"precipitation_rate": 0.0, "fog_density": 0.8, "wind_speed": 0.5},
    "snow":  {"precipitation_rate": 3.0, "fog_density": 0.3, "wind_speed": 1.5},
}

# time_of_day → (ambient_r, ambient_g, ambient_b, ambient_a)
TIME_OF_DAY_TO_AMBIENT: Dict[str, Tuple[float, float, float, float]] = {
    "day":   (0.8, 0.8, 0.8, 1.0),
    "dusk":  (0.5, 0.35, 0.2, 1.0),
    "night": (0.1, 0.1, 0.15, 1.0),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heading_to_radians(degrees: float) -> float:
    """Convert a heading in degrees to radians."""
    return math.radians(degrees)


def _world_file(environment: Environment) -> str:
    """Return the Gazebo world file name for the given environment."""
    return ROAD_TYPE_TO_WORLD.get(environment.road_type, "default.world")


def _weather_params(environment: Environment) -> Dict[str, float]:
    """Return Gazebo weather plugin parameters for the given environment."""
    return WEATHER_TO_GAZEBO_PARAMS.get(
        environment.weather, WEATHER_TO_GAZEBO_PARAMS["clear"]
    )


def _ambient_light(environment: Environment) -> Tuple[float, float, float, float]:
    """Return RGBA ambient light values for the given time of day."""
    return TIME_OF_DAY_TO_AMBIENT.get(
        environment.time_of_day, TIME_OF_DAY_TO_AMBIENT["day"]
    )


def _actor_model(actor: Actor) -> str:
    """Return the Gazebo model name for the given actor type."""
    return ACTOR_TYPE_TO_MODEL.get(actor.actor_type, "vehicle_box")


# ---------------------------------------------------------------------------
# Spawn-node helpers (produce plain strings, no indentation applied here)
# ---------------------------------------------------------------------------


def _spawn_node_lines(
    name: str,
    model: str,
    x: float,
    y: float,
    yaw_rad: float,
    vx: float,
    namespace: str,
) -> List[str]:
    """
    Return the lines for a pair of ROS 2 Node(...) calls that spawn *name*
    and set its initial velocity.  Each line is NOT indented — callers indent.
    """
    return [
        f"# --- {name} ---",
        f'spawn_{name} = Node(',
        f'    package="gazebo_ros",',
        f'    executable="spawn_entity.py",',
        f'    name="spawn_{name}",',
        f'    namespace="{namespace}",',
        f'    arguments=[',
        f'        "-entity", "{name}",',
        f'        "-database", "{model}",',
        f'        "-x", "{x:.4f}",',
        f'        "-y", "{y:.4f}",',
        f'        "-z", "0.0000",',
        f'        "-Y", "{yaw_rad:.6f}",',
        f'    ],',
        f'    output="screen",',
        f')',
        f'init_vel_{name} = Node(',
        f'    package="ros2_initial_velocity",',
        f'    executable="set_twist",',
        f'    name="init_vel_{name}",',
        f'    namespace="{namespace}",',
        f'    parameters=[{{"linear_x": {vx:.4f}, "entity": "{name}"}}],',
        f')',
        f'',
    ]


def _indent(lines: List[str], spaces: int = 4) -> str:
    """Join *lines* with newlines, indenting each non-empty line by *spaces*."""
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else "" for line in lines)


def _node_names(name: str) -> str:
    """Return the two variable names for a spawn entity as a list-entry pair."""
    return f"        spawn_{name},\n        init_vel_{name},"


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------


def export_to_ros_launch(config: ScenarioConfig) -> str:
    """
    Generate a ROS 2 Python launch file from a ScenarioConfig.

    Parameters
    ----------
    config:
        A fully validated ScenarioConfig instance.

    Returns
    -------
    str
        The complete content of a ROS 2 Python launch file (scenario_launch.py).
    """
    env = config.environment
    world = _world_file(env)
    wx = _weather_params(env)
    ambient = _ambient_light(env)

    # Build node definition lines (indented 4 spaces = inside function body)
    all_node_lines: List[str] = []

    # ego vehicle
    ego_lines = _spawn_node_lines(
        name="ego_vehicle",
        model="ego_vehicle",
        x=0.0,
        y=0.0,
        yaw_rad=0.0,
        vx=config.ego_vehicle_speed_mps,
        namespace="ego",
    )
    all_node_lines.extend(ego_lines)

    # actors
    for i, actor in enumerate(config.actors):
        lines = _spawn_node_lines(
            name=f"actor_{i}",
            model=_actor_model(actor),
            x=actor.start_position[0],
            y=actor.start_position[1],
            yaw_rad=_heading_to_radians(actor.heading_degrees),
            vx=actor.speed_mps,
            namespace=f"actor_{i}",
        )
        all_node_lines.extend(lines)

    node_definitions = _indent(all_node_lines, spaces=4)

    # Build the return list entries
    return_entries = ["    return LaunchDescription([",
                      "        gazebo,",
                      "        weather_config,",
                      "        spawn_ego_vehicle,",
                      "        init_vel_ego_vehicle,"]
    for i in range(len(config.actors)):
        return_entries.append(f"        spawn_actor_{i},")
        return_entries.append(f"        init_vel_actor_{i},")
    return_entries.append("    ])")
    return_block = "\n".join(return_entries)

    launch_file = (
        '"""\n'
        f'Auto-generated ROS 2 Python launch file.\n'
        f'Scenario   : {config.scenario_id}\n'
        f'Description: {config.description}\n'
        f'Duration   : {config.duration_seconds}s\n'
        'Generated by ScenarioGen ros_export.py — do not edit manually.\n'
        '"""\n'
        '\n'
        'import os\n'
        '\n'
        'from ament_index_python.packages import get_package_share_directory\n'
        'from launch import LaunchDescription\n'
        'from launch.actions import IncludeLaunchDescription\n'
        'from launch.launch_description_sources import PythonLaunchDescriptionSource\n'
        'from launch_ros.actions import Node\n'
        '\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# World / weather / lighting constants (derived from ScenarioConfig)\n'
        '# ---------------------------------------------------------------------------\n'
        '\n'
        f'WORLD_FILE = "{world}"\n'
        f'WEATHER_PRECIPITATION_RATE = {wx["precipitation_rate"]:.2f}\n'
        f'WEATHER_FOG_DENSITY = {wx["fog_density"]:.2f}\n'
        f'WEATHER_WIND_SPEED = {wx["wind_speed"]:.2f}\n'
        f'AMBIENT_LIGHT = ({ambient[0]:.2f}, {ambient[1]:.2f}, {ambient[2]:.2f}, {ambient[3]:.2f})\n'
        '\n'
        '\n'
        'def generate_launch_description() -> LaunchDescription:\n'
        '    """Entry point required by ROS 2 launch system."""\n'
        '\n'
        '    gazebo_ros_pkg = get_package_share_directory("gazebo_ros")\n'
        '    world_path = os.path.join(\n'
        '        get_package_share_directory("scenariogen_worlds"),\n'
        '        "worlds",\n'
        '        WORLD_FILE,\n'
        '    )\n'
        '\n'
        '    gazebo = IncludeLaunchDescription(\n'
        '        PythonLaunchDescriptionSource(\n'
        '            os.path.join(gazebo_ros_pkg, "launch", "gazebo.launch.py")\n'
        '        ),\n'
        '        launch_arguments={\n'
        '            "world": world_path,\n'
        '            "verbose": "false",\n'
        '        }.items(),\n'
        '    )\n'
        '\n'
        '    weather_config = Node(\n'
        '        package="gazebo_ros",\n'
        '        executable="set_weather_params",\n'
        '        name="weather_configurator",\n'
        '        parameters=[{\n'
        '            "precipitation_rate": WEATHER_PRECIPITATION_RATE,\n'
        '            "fog_density": WEATHER_FOG_DENSITY,\n'
        '            "wind_speed": WEATHER_WIND_SPEED,\n'
        '            "ambient_r": AMBIENT_LIGHT[0],\n'
        '            "ambient_g": AMBIENT_LIGHT[1],\n'
        '            "ambient_b": AMBIENT_LIGHT[2],\n'
        '            "ambient_a": AMBIENT_LIGHT[3],\n'
        '        }],\n'
        '    )\n'
        '\n'
        + node_definitions + '\n'
        '\n'
        + return_block + '\n'
    )

    return launch_file


# ---------------------------------------------------------------------------
# File-writing function
# ---------------------------------------------------------------------------


def write_ros_launch(config: ScenarioConfig, output_dir: str = ".") -> str:
    """
    Write the ROS 2 launch file to *output_dir* and return the output path.

    Parameters
    ----------
    config:
        A fully validated ScenarioConfig instance.
    output_dir:
        Directory where ``scenario_launch.py`` will be written. Created if it
        does not already exist.

    Returns
    -------
    str
        Absolute path of the written launch file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    launch_content = export_to_ros_launch(config)
    out_path = os.path.join(output_dir, "scenario_launch.py")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(launch_content)
    return os.path.abspath(out_path)