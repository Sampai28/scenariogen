"""
tests/test_ros_export.py

Unit, integration, and comparison tests for src/ros_export.py.

Run with:
    python -m pytest tests/test_ros_export.py -v
"""

from __future__ import annotations

import ast
import importlib
import math
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schema import Actor, Environment, ScenarioConfig  # noqa: E402
from src.ros_export import (  # noqa: E402
    ACTOR_TYPE_TO_MODEL,
    ROAD_TYPE_TO_WORLD,
    TIME_OF_DAY_TO_AMBIENT,
    WEATHER_TO_GAZEBO_PARAMS,
    _heading_to_radians,
    export_to_ros_launch,
    write_ros_launch,
)

# ---------------------------------------------------------------------------
# Fixtures / factory helpers
# ---------------------------------------------------------------------------


def make_actor(
    actor_type: str = "vehicle",
    x: float = 10.0,
    y: float = 5.0,
    heading: float = 0.0,
    speed: float = 5.0,
    behavior: str = "driving straight",
) -> Actor:
    return Actor(
        actor_type=actor_type,
        start_position=(x, y),
        heading_degrees=heading,
        speed_mps=speed,
        behavior=behavior,
    )


def make_env(
    road_type: str = "intersection",
    weather: str = "clear",
    time_of_day: str = "day",
) -> Environment:
    return Environment(road_type=road_type, weather=weather, time_of_day=time_of_day)


def make_config(
    actors: list[Actor] | None = None,
    road_type: str = "intersection",
    weather: str = "clear",
    time_of_day: str = "day",
    ego_speed: float = 10.0,
    description: str = "test scenario",
    duration: float = 10.0,
) -> ScenarioConfig:
    return ScenarioConfig(
        description=description,
        ego_vehicle_speed_mps=ego_speed,
        actors=actors or [make_actor()],
        environment=make_env(road_type=road_type, weather=weather, time_of_day=time_of_day),
        duration_seconds=duration,
    )


def make_highway_config(actors: list[Actor] | None = None) -> ScenarioConfig:
    """Highway requires ego speed >= 20 and no pedestrians."""
    safe_actors = actors or [make_actor(actor_type="vehicle")]
    return make_config(actors=safe_actors, road_type="highway", ego_speed=25.0)


# ===========================================================================
# 1. UNIT TESTS
# ===========================================================================


class TestValidPythonSyntax:
    """Generated launch file must compile without syntax errors."""

    def test_basic_config_produces_valid_python(self):
        config = make_config()
        source = export_to_ros_launch(config)
        tree = ast.parse(source)
        assert tree is not None

    def test_all_actor_types_produce_valid_python(self):
        # Pedestrians can't be on highway, use intersection
        actors = [
            make_actor(actor_type="vehicle", x=0.0),
            make_actor(actor_type="pedestrian", x=5.0, speed=1.0),
            make_actor(actor_type="cyclist", x=10.0, speed=5.0),
            make_actor(actor_type="motorcycle", x=15.0),
        ]
        config = make_config(actors=actors, road_type="intersection")
        source = export_to_ros_launch(config)
        ast.parse(source)


class TestRoadTypeMapping:
    """Every road_type must map to the correct .world file."""

    @pytest.mark.parametrize("road_type", list(ROAD_TYPE_TO_WORLD.keys()))
    def test_road_type_maps_to_world_file(self, road_type: str):
        if road_type == "highway":
            config = make_highway_config()
        else:
            config = make_config(road_type=road_type)
        source = export_to_ros_launch(config)
        expected = ROAD_TYPE_TO_WORLD[road_type]
        assert expected in source

    def test_all_four_road_types_covered(self):
        assert set(ROAD_TYPE_TO_WORLD.keys()) == {
            "intersection", "highway", "residential", "parking_lot"
        }


class TestWeatherMapping:
    """Every weather condition must produce valid Gazebo plugin parameters."""

    @pytest.mark.parametrize("weather", list(WEATHER_TO_GAZEBO_PARAMS.keys()))
    def test_weather_produces_valid_params(self, weather: str):
        params = WEATHER_TO_GAZEBO_PARAMS[weather]
        assert "precipitation_rate" in params
        assert "fog_density" in params
        assert "wind_speed" in params
        for v in params.values():
            assert isinstance(v, (int, float))

    @pytest.mark.parametrize("weather", list(WEATHER_TO_GAZEBO_PARAMS.keys()))
    def test_weather_params_appear_in_launch(self, weather: str):
        config = make_config(weather=weather)
        source = export_to_ros_launch(config)
        params = WEATHER_TO_GAZEBO_PARAMS[weather]
        assert f"{params['fog_density']:.2f}" in source

    def test_all_four_weather_types_covered(self):
        assert set(WEATHER_TO_GAZEBO_PARAMS.keys()) == {"clear", "rain", "fog", "snow"}


class TestTimeOfDayMapping:
    """Every time_of_day value must map to a distinct lighting config."""

    @pytest.mark.parametrize("tod", list(TIME_OF_DAY_TO_AMBIENT.keys()))
    def test_time_of_day_maps_to_lighting(self, tod: str):
        rgba = TIME_OF_DAY_TO_AMBIENT[tod]
        assert len(rgba) == 4
        for channel in rgba:
            assert 0.0 <= channel <= 1.0

    def test_all_three_times_covered(self):
        assert set(TIME_OF_DAY_TO_AMBIENT.keys()) == {"day", "dusk", "night"}

    @pytest.mark.parametrize("tod", list(TIME_OF_DAY_TO_AMBIENT.keys()))
    def test_ambient_values_appear_in_launch(self, tod: str):
        config = make_config(time_of_day=tod)
        source = export_to_ros_launch(config)
        r = TIME_OF_DAY_TO_AMBIENT[tod][0]
        assert f"{r:.2f}" in source


class TestHeadingConversion:
    """heading_degrees must be accurately converted to radians."""

    @pytest.mark.parametrize(
        "degrees, expected_rad",
        [
            (0.0,   0.0),
            (90.0,  math.pi / 2),
            (180.0, math.pi),
            (270.0, 3 * math.pi / 2),
            (45.0,  math.pi / 4),
        ],
    )
    def test_heading_to_radians(self, degrees: float, expected_rad: float):
        assert math.isclose(_heading_to_radians(degrees), expected_rad, rel_tol=1e-9)

    def test_heading_appears_in_launch_file(self):
        actor = make_actor(heading=90.0)
        config = make_config(actors=[actor])
        source = export_to_ros_launch(config)
        assert f"{math.radians(90.0):.6f}" in source


class TestActorTypeMapping:
    """All actor types must map to Gazebo model names and appear in output."""

    @pytest.mark.parametrize("actor_type", list(ACTOR_TYPE_TO_MODEL.keys()))
    def test_actor_type_maps_to_model(self, actor_type: str):
        model = ACTOR_TYPE_TO_MODEL[actor_type]
        assert isinstance(model, str) and len(model) > 0

    def test_all_four_actor_types_covered(self):
        assert set(ACTOR_TYPE_TO_MODEL.keys()) == {
            "vehicle", "pedestrian", "cyclist", "motorcycle"
        }

    @pytest.mark.parametrize("actor_type", list(ACTOR_TYPE_TO_MODEL.keys()))
    def test_model_name_appears_in_launch(self, actor_type: str):
        speed = 1.0 if actor_type == "pedestrian" else 5.0
        actor = make_actor(actor_type=actor_type, speed=speed)
        config = make_config(actors=[actor], road_type="intersection")
        source = export_to_ros_launch(config)
        assert ACTOR_TYPE_TO_MODEL[actor_type] in source


class TestEgoVehicleSpawn:
    """Ego vehicle must always be present in the launch output."""

    def test_ego_vehicle_in_launch(self):
        config = make_config()
        source = export_to_ros_launch(config)
        assert "ego_vehicle" in source

    def test_ego_speed_in_launch(self):
        config = make_config(ego_speed=15.0)
        source = export_to_ros_launch(config)
        assert "15.0000" in source


class TestActorCount:
    """Actor count in launch file must match ScenarioConfig.actors length."""

    @pytest.mark.parametrize("n_actors", [1, 3, 6])
    def test_actor_count_matches(self, n_actors: int):
        actors = [make_actor(x=float(i * 5)) for i in range(n_actors)]
        config = make_config(actors=actors)
        source = export_to_ros_launch(config)
        for i in range(n_actors):
            assert f"actor_{i}" in source


class TestEdgeCases:
    """Boundary actor counts and speeds."""

    def test_single_actor(self):
        config = make_config(actors=[make_actor()])
        source = export_to_ros_launch(config)
        ast.parse(source)
        assert "actor_0" in source

    def test_max_actors_six(self):
        actors = [make_actor(x=float(i * 3)) for i in range(6)]
        config = make_config(actors=actors)
        source = export_to_ros_launch(config)
        ast.parse(source)
        assert "actor_5" in source

    def test_zero_ego_speed(self):
        config = make_config(ego_speed=0.0)
        source = export_to_ros_launch(config)
        ast.parse(source)
        assert "0.0000" in source

    def test_max_ego_speed_fifty(self):
        config = make_config(ego_speed=50.0)
        source = export_to_ros_launch(config)
        ast.parse(source)
        assert "50.0000" in source


# ===========================================================================
# 2. INTEGRATION TESTS
# ===========================================================================


class TestIntegration:
    """Full pipeline: ScenarioConfig → launch string → validate."""

    def test_non_highway_combos_all_valid(self):
        """All non-highway road types with all actor types, weather, and times."""
        non_highway = [rt for rt in ROAD_TYPE_TO_WORLD if rt != "highway"]
        actors = [
            make_actor(actor_type="vehicle",    x=0.0,  speed=5.0),
            make_actor(actor_type="pedestrian", x=5.0,  speed=1.0),
            make_actor(actor_type="cyclist",    x=10.0, speed=5.0),
            make_actor(actor_type="motorcycle", x=15.0, speed=5.0),
        ]
        for road_type in non_highway:
            for weather in WEATHER_TO_GAZEBO_PARAMS:
                for tod in TIME_OF_DAY_TO_AMBIENT:
                    config = make_config(
                        actors=actors,
                        road_type=road_type,
                        weather=weather,
                        time_of_day=tod,
                    )
                    source = export_to_ros_launch(config)
                    assert source
                    ast.parse(source)

    def test_highway_combos_all_valid(self):
        """Highway combos with vehicle/cyclist/motorcycle only (no pedestrians)."""
        actors = [
            make_actor(actor_type="vehicle",    x=0.0,  speed=25.0),
            make_actor(actor_type="cyclist",    x=10.0, speed=5.0),
            make_actor(actor_type="motorcycle", x=20.0, speed=25.0),
        ]
        for weather in WEATHER_TO_GAZEBO_PARAMS:
            for tod in TIME_OF_DAY_TO_AMBIENT:
                config = make_config(
                    actors=actors,
                    road_type="highway",
                    weather=weather,
                    time_of_day=tod,
                    ego_speed=25.0,
                )
                source = export_to_ros_launch(config)
                assert source
                ast.parse(source)

    def test_output_contains_ros2_launch_patterns(self):
        config = make_config()
        source = export_to_ros_launch(config)
        assert "from launch import LaunchDescription" in source
        assert "from launch_ros.actions import Node" in source
        assert "generate_launch_description()" in source
        assert "gazebo_ros" in source

    def test_spawn_entity_pattern_present(self):
        config = make_config()
        source = export_to_ros_launch(config)
        assert "spawn_entity.py" in source

    def test_write_ros_launch_creates_file(self):
        config = make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_ros_launch(config, output_dir=tmpdir)
            assert os.path.isfile(path)
            assert path.endswith("scenario_launch.py")
            ast.parse(open(path, encoding="utf-8").read())

    def test_write_ros_launch_creates_nested_output_dir(self):
        config = make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            path = write_ros_launch(config, output_dir=nested)
            assert os.path.isfile(path)


# ===========================================================================
# 3. COMPARISON TESTS  (ROS vs CARLA)
# ===========================================================================


class TestComparisonWithCarla:
    """Same ScenarioConfig should export to both CARLA and ROS without error."""

    def _make_full_config(self) -> ScenarioConfig:
        actors = [
            make_actor(actor_type="vehicle",    x=10.0, y=0.0,  speed=10.0),
            make_actor(actor_type="cyclist",    x=5.0,  y=5.0,  speed=5.0),
            make_actor(actor_type="motorcycle", x=0.0,  y=-8.0, speed=10.0),
        ]
        return make_config(
            actors=actors,
            road_type="residential",
            weather="rain",
            time_of_day="dusk",
            ego_speed=10.0,
        )

    def test_both_exports_succeed(self):
        from src.carla_export import export_to_carla  # noqa: PLC0415

        config = self._make_full_config()
        carla_out = export_to_carla(config)
        ros_out = export_to_ros_launch(config)
        assert carla_out, "CARLA export returned empty string"
        assert ros_out, "ROS export returned empty string"

    def test_actor_count_consistent_ros(self):
        config = self._make_full_config()
        source = export_to_ros_launch(config)
        for i in range(len(config.actors)):
            assert f"actor_{i}" in source

    def test_ego_speed_in_ros_output(self):
        config = self._make_full_config()
        source = export_to_ros_launch(config)
        assert f"{config.ego_vehicle_speed_mps:.4f}" in source

    def test_carla_export_unchanged_after_ros_import(self):
        from src.carla_export import export_to_carla  # noqa: PLC0415
        import src.ros_export as re_mod  # noqa: PLC0415

        config = self._make_full_config()
        out1 = export_to_carla(config)
        importlib.reload(re_mod)
        out2 = export_to_carla(config)
        assert out1 == out2