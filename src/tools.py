"""
LangChain tools for the AV scenario generation pipeline.

Tool pipeline:
  1. ScenarioParser    - extract actors and environment from NL input
  2. ConstraintChecker - validate and fix physical plausibility
  3. ConfigGenerator   - assemble final Pydantic-validated JSON
"""

import json
import uuid
from langchain_core.tools import tool
from pydantic import ValidationError
from src.schema import Actor, Environment, ScenarioConfig


@tool
def parse_scenario(natural_language_description: str) -> str:
    """
    Extract structured actors and environment data from a natural language
    AV scenario description. Returns a JSON string with actors and
    environment keys ready for constraint checking.

    Input: raw NL description e.g. 'a pedestrian jaywalks at night in rain'
    Output: JSON string with extracted scenario components
    """
    extraction_template = {
        "instruction": "parse_complete",
        "input_description": natural_language_description,
        "expected_output_format": {
            "actors": [
                {
                    "actor_type": "<vehicle|pedestrian|cyclist|motorcycle>",
                    "start_position": [0.0, 0.0],
                    "heading_degrees": 90.0,
                    "speed_mps": 0.0,
                    "behavior": "<description>"
                }
            ],
            "environment": {
                "time_of_day": "<day|dusk|night>",
                "weather": "<clear|rain|fog|snow>",
                "road_type": "<intersection|highway|residential|parking_lot>"
            },
            "ego_vehicle_speed_mps": 0.0,
            "duration_seconds": 10.0
        }
    }
    return json.dumps(extraction_template)


@tool
def check_constraints(scenario_json: str | dict) -> str:
    """
    Validate physical plausibility of a scenario JSON string.
    Checks speed limits per actor type, actor count, and road type rules.
    Returns a JSON string with valid (bool), warnings (list),
    and fixed_scenario (dict with auto-corrections applied).

    Input: JSON string with actors, environment, ego_vehicle_speed_mps,
           duration_seconds fields
    Output: JSON string with validation results and corrections
    """
    warnings = []

    try:
        if isinstance(scenario_json, dict):
            data = scenario_json
        else:
            data = json.loads(scenario_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "warnings": ["Invalid JSON: " + str(e)], "fixed_scenario": None})

    actors = data.get("actors", [])
    env = data.get("environment", {})
    ego_speed = data.get("ego_vehicle_speed_mps", 10.0)
    duration = data.get("duration_seconds", 10.0)

    speed_limits = {
        "vehicle": 50.0,
        "motorcycle": 50.0,
        "cyclist": 12.0,
        "pedestrian": 3.0,
    }

    fixed_actors = []
    for i, actor in enumerate(actors):
        actor = dict(actor)
        atype = actor.get("actor_type", "vehicle")
        limit = speed_limits.get(atype, 50.0)
        if actor.get("speed_mps", 0) > limit:
            warnings.append(
                "Actor " + str(i) + " (" + atype + ") speed " +
                str(actor["speed_mps"]) + " m/s exceeds max " +
                str(limit) + " m/s, clamped"
            )
            actor["speed_mps"] = limit
        h = actor.get("heading_degrees", 90.0)
        if not (0 <= h < 360):
            actor["heading_degrees"] = h % 360
            warnings.append("Actor " + str(i) + " heading clamped to " + str(actor["heading_degrees"]))
        fixed_actors.append(actor)

    if len(fixed_actors) == 0:
        warnings.append("No actors found, adding a default vehicle")
        fixed_actors.append({
            "actor_type": "vehicle",
            "start_position": [10.0, 0.0],
            "heading_degrees": 90.0,
            "speed_mps": 10.0,
            "behavior": "driving straight"
        })
    if len(fixed_actors) > 6:
        warnings.append("Too many actors (" + str(len(fixed_actors)) + "), truncated to 6")
        fixed_actors = fixed_actors[:6]

    road_type = env.get("road_type", "intersection")
    if road_type == "highway":
        if ego_speed < 20:
            warnings.append("Highway ego speed too low, set to 25 m/s")
            ego_speed = 25.0
        original_count = len(fixed_actors)
        fixed_actors = [a for a in fixed_actors if a.get("actor_type") != "pedestrian"]
        if len(fixed_actors) < original_count:
            warnings.append("Removed pedestrians from highway scenario")

    if duration <= 0 or duration > 60:
        warnings.append("Duration out of range, set to 10s")
        duration = 10.0

    if ego_speed < 0 or ego_speed > 50:
        warnings.append("Ego speed out of range, set to 10 m/s")
        ego_speed = 10.0

    fixed = {
        "actors": fixed_actors,
        "environment": env,
        "ego_vehicle_speed_mps": ego_speed,
        "duration_seconds": duration,
    }

    return json.dumps({
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "fixed_scenario": fixed
    })


@tool
def generate_config(checked_scenario_json: str | dict) -> str:
    """
    Assemble a final Pydantic-validated ScenarioConfig from a checked
    scenario JSON string (output of check_constraints).
    Returns the final validated scenario as a JSON string, or an error message.

    Input: JSON string containing fixed_scenario and description fields
    Output: Final validated ScenarioConfig as JSON string
    """
    try:
        if isinstance(checked_scenario_json, dict):
            data = checked_scenario_json
        else:
            data = json.loads(checked_scenario_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": "Invalid JSON input: " + str(e)})

    scenario = data.get("fixed_scenario") or data
    description = data.get("description") or scenario.get("description", "Generated AV scenario")

    try:
        config = ScenarioConfig(
            scenario_id=str(uuid.uuid4())[:8],
            description=description,
            ego_vehicle_speed_mps=scenario.get("ego_vehicle_speed_mps", 10.0),
            actors=[Actor(**a) for a in scenario.get("actors", [])],
            environment=Environment(**scenario.get("environment", {})),
            duration_seconds=scenario.get("duration_seconds", 10.0),
        )
        return config.model_dump_json(indent=2)

    except ValidationError as e:
        return json.dumps({"error": "Validation failed: " + str(e)})
    except Exception as e:
        return json.dumps({"error": "Config generation failed: " + str(e)})