"""
ScenarioGen — LangChain tools for the AV scenario generation agent.

Tool pipeline:
  1. ScenarioParser   → extract actors + environment from NL input
  2. ConstraintChecker → validate physical plausibility
  3. ConfigGenerator  → assemble final Pydantic-validated JSON
"""

import json
import uuid
from langchain_core.tools import tool
from pydantic import ValidationError
from src.schema import Actor, Environment, ScenarioConfig


# ── Tool 1: ScenarioParser ────────────────────────────────────────────────────

@tool
def parse_scenario(natural_language_description: str) -> str:
    """
    Extract structured actors and environment data from a natural language
    AV scenario description. Returns a JSON string with 'actors' and
    'environment' keys ready for constraint checking.

    Input: raw NL description e.g. 'a pedestrian jaywalks at night in rain'
    Output: JSON string with extracted scenario components
    """
    # Return a structured extraction prompt result as JSON template
    # The agent's LLM will fill this in via its reasoning
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


# ── Tool 2: ConstraintChecker ─────────────────────────────────────────────────

@tool
def check_constraints(scenario_json: str | dict) -> str:
    """
    Validate physical plausibility of a scenario JSON string.
    Checks speed limits per actor type, actor count, road type rules.
    Returns a JSON string with 'valid' (bool), 'warnings' (list), 
    and 'fixed_scenario' (dict with auto-corrections applied).

    Input: JSON string with actors, environment, ego_vehicle_speed_mps,
           duration_seconds fields
    Output: JSON string with validation results and corrections
    """
    warnings = []
    
    try:
        # Accept either a JSON string or already-parsed dict
        if isinstance(scenario_json, dict):
            data = scenario_json
        else:
            data = json.loads(scenario_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "warnings": [f"Invalid JSON: {e}"], "fixed_scenario": None})

    actors = data.get("actors", [])
    env = data.get("environment", {})
    ego_speed = data.get("ego_vehicle_speed_mps", 10.0)
    duration = data.get("duration_seconds", 10.0)

    # Speed limits per actor type
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
                f"Actor {i} ({atype}) speed {actor['speed_mps']} m/s "
                f"exceeds max {limit} m/s — clamped to {limit}"
            )
            actor["speed_mps"] = limit
        # Clamp heading to [0, 360)
        h = actor.get("heading_degrees", 90.0)
        if not (0 <= h < 360):
            actor["heading_degrees"] = h % 360
            warnings.append(f"Actor {i} heading clamped to {actor['heading_degrees']}")
        fixed_actors.append(actor)

    # Actor count
    if len(fixed_actors) == 0:
        warnings.append("No actors found — adding a default vehicle")
        fixed_actors.append({
            "actor_type": "vehicle", "start_position": [10.0, 0.0],
            "heading_degrees": 90.0, "speed_mps": 10.0,
            "behavior": "driving straight"
        })
    if len(fixed_actors) > 6:
        warnings.append(f"Too many actors ({len(fixed_actors)}) — truncated to 6")
        fixed_actors = fixed_actors[:6]

    # Highway rules
    road_type = env.get("road_type", "intersection")
    if road_type == "highway":
        if ego_speed < 20:
            warnings.append(f"Highway ego speed {ego_speed} too low — set to 25 m/s")
            ego_speed = 25.0
        fixed_actors = [
            a for a in fixed_actors if a.get("actor_type") != "pedestrian"
        ]
        if len(fixed_actors) < len(actors):
            warnings.append("Removed pedestrians from highway scenario")

    # Duration clamp
    if duration <= 0 or duration > 60:
        warnings.append(f"Duration {duration}s out of range — set to 10s")
        duration = 10.0

    # Ego speed clamp
    if ego_speed < 0 or ego_speed > 50:
        warnings.append(f"Ego speed {ego_speed} out of range — set to 10 m/s")
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


# ── Tool 3: ConfigGenerator ───────────────────────────────────────────────────

@tool
def generate_config(checked_scenario_json: str) -> str:
    """
    Assemble a final, Pydantic-validated ScenarioConfig from a checked
    scenario JSON string (output of check_constraints).
    Returns the final validated scenario as a JSON string, or an error message.

    Input: JSON string — output from check_constraints tool
           Must contain 'fixed_scenario' and original 'description' fields
    Output: Final validated ScenarioConfig as JSON string
    """
    try:
        # Accept either a JSON string or already-parsed dict
        if isinstance(checked_scenario_json, dict):
            data = checked_scenario_json
        else:
            data = json.loads(checked_scenario_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON input: {e}"})

    # Accept either raw scenario or output from check_constraints
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
        return json.dumps({"error": f"Validation failed: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Config generation failed: {e}"})