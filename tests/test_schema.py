"""
Tests for ScenarioConfig schema validation.
Run with: pytest tests/test_schema.py -v
"""

import pytest
from pydantic import ValidationError
from src.schema import Actor, Environment, ScenarioConfig


# Helpers

def make_env(road_type="intersection", weather="clear", time="day"):
    return Environment(road_type=road_type, weather=weather, time_of_day=time)

def make_actor(actor_type="vehicle", speed=10.0, heading=90.0, behavior="driving straight"):
    return Actor(
        actor_type=actor_type,
        start_position=(0.0, 5.0),
        heading_degrees=heading,
        speed_mps=speed,
        behavior=behavior,
    )

def make_scenario(**overrides):
    defaults = dict(
        description="A car approaches an intersection",
        ego_vehicle_speed_mps=10.0,
        actors=[make_actor()],
        environment=make_env(),
        duration_seconds=10.0,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


# Valid Scenarios

def test_valid_basic_scenario():
    s = make_scenario()
    assert s.description == "A car approaches an intersection"
    assert len(s.actors) == 1

def test_scenario_id_auto_generated():
    s = make_scenario()
    assert len(s.scenario_id) == 8

def test_multiple_actors():
    actors = [
        make_actor("vehicle", speed=10.0),
        make_actor("pedestrian", speed=1.2, behavior="crossing street"),
        make_actor("cyclist", speed=5.0, behavior="riding in bike lane"),
    ]
    s = make_scenario(actors=actors)
    assert len(s.actors) == 3

def test_night_rain_scenario():
    s = make_scenario(environment=make_env(weather="rain", time="night"))
    assert s.environment.weather == "rain"


# ── Invalid scenarios ─────────────────────────────────────────────────────────

def test_pedestrian_too_fast():
    with pytest.raises(ValidationError, match="speed"):
        make_actor(actor_type="pedestrian", speed=10.0)  # max is 3 m/s

def test_cyclist_too_fast():
    with pytest.raises(ValidationError, match="speed"):
        make_actor(actor_type="cyclist", speed=20.0)  # max is 12 m/s

def test_vehicle_too_fast():
    with pytest.raises(ValidationError, match="speed"):
        make_actor(actor_type="vehicle", speed=60.0)  # max is 50 m/s

def test_negative_speed():
    with pytest.raises(ValidationError):
        make_actor(speed=-5.0)

def test_invalid_heading():
    with pytest.raises(ValidationError):
        make_actor(heading=400.0)

def test_too_many_actors():
    with pytest.raises(ValidationError):
        make_scenario(actors=[make_actor() for _ in range(7)])  # max is 6

def test_no_actors():
    with pytest.raises(ValidationError):
        make_scenario(actors=[])

def test_pedestrian_on_highway():
    with pytest.raises(ValidationError, match="Pedestrians cannot appear"):
        make_scenario(
            environment=make_env(road_type="highway"),
            ego_vehicle_speed_mps=25.0,
            actors=[make_actor("pedestrian", speed=1.5, behavior="walking")],
        )

def test_highway_too_slow():
    with pytest.raises(ValidationError, match="highway"):
        make_scenario(
            environment=make_env(road_type="highway"),
            ego_vehicle_speed_mps=5.0,  # too slow for highway
            actors=[make_actor("vehicle", speed=10.0)],
        )

def test_empty_description():
    with pytest.raises(ValidationError):
        make_scenario(description="   ")