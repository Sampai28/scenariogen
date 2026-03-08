"""
ScenarioGen — Direct JSON generation using Groq LLM + Pydantic validation.
No tool calling required — works with smaller models like llama-3.1-8b-instant.
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.schema import ScenarioConfig
from src.tools import check_constraints

load_dotenv()

# ── Prompts ───────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are an AV scenario engineer. Convert natural language into a JSON simulation config.

Return ONLY a valid JSON object with this exact structure — no explanation, no markdown, no backticks:
{
  "scenario_id": "auto",
  "description": "<original input>",
  "ego_vehicle_speed_mps": <float>,
  "duration_seconds": <float 5-30>,
  "actors": [
    {
      "actor_type": "<vehicle|pedestrian|cyclist|motorcycle>",
      "start_position": [<x float>, <y float>],
      "heading_degrees": <float 0-359>,
      "speed_mps": <float>,
      "behavior": "<description>"
    }
  ],
  "environment": {
    "time_of_day": "<day|dusk|night>",
    "weather": "<clear|rain|fog|snow>",
    "road_type": "<intersection|highway|residential|parking_lot>"
  }
}

Speed limits: pedestrian<=3, cyclist<=12, vehicle/motorcycle<=50.
Highway ego speed >= 20. No pedestrians on highway. Actors: 1-6. Duration: 5-30s.
Position actors 5-30 meters apart from origin (ego is at 0,0)."""

FEW_SHOT_SUFFIX = """

Examples:

Input: "pedestrian jaywalks at night in rain"
Output: {"scenario_id":"auto","description":"pedestrian jaywalks at night in rain","ego_vehicle_speed_mps":10.0,"duration_seconds":8.0,"actors":[{"actor_type":"pedestrian","start_position":[8.0,0.0],"heading_degrees":270.0,"speed_mps":1.2,"behavior":"jaywalking across road"}],"environment":{"time_of_day":"night","weather":"rain","road_type":"residential"}}

Input: "cyclist cuts across intersection while car approaches"
Output: {"scenario_id":"auto","description":"cyclist cuts across intersection while car approaches","ego_vehicle_speed_mps":12.0,"duration_seconds":10.0,"actors":[{"actor_type":"cyclist","start_position":[0.0,15.0],"heading_degrees":180.0,"speed_mps":5.0,"behavior":"cutting across intersection"},{"actor_type":"vehicle","start_position":[20.0,0.0],"heading_degrees":90.0,"speed_mps":10.0,"behavior":"approaching intersection"}],"environment":{"time_of_day":"day","weather":"clear","road_type":"intersection"}}"""

COT_SUFFIX = """

Before writing JSON, think through:
1. What actors are present and what type?
2. What speed fits each actor's behavior?
3. What environment — time, weather, road type?
4. Where should actors be positioned (5-30m apart)?
Then output ONLY the JSON."""


# ── LLM factory ───────────────────────────────────────────────────────────────

def build_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


# ── Core generation ───────────────────────────────────────────────────────────

def _invoke_with_retry(llm, messages, max_retries=3):
    """Invoke LLM with rate-limit retry."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 20 * (attempt + 1)
                print(f"[Agent] Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def _extract_json(text: str) -> dict | None:
    """Extract first valid JSON object from text."""
    # Strip markdown code fences if present
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _apply_constraints(data) -> dict:
    """Run constraint checker and return fixed scenario data."""
    payload = data if isinstance(data, str) else json.dumps(data)
    result = json.loads(check_constraints.invoke({"scenario_json": payload}))
    fixed = result.get("fixed_scenario") or data
    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"[Constraints] {w}")
    return fixed


def _validate(data: dict, description: str) -> ScenarioConfig | None:
    """Try to build a validated ScenarioConfig from raw dict."""
    import uuid
    try:
        data["scenario_id"] = str(uuid.uuid4())[:8]
        data["description"] = description
        return ScenarioConfig(**data)
    except Exception as e:
        print(f"[Validation] Failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_scenario(description: str, strategy: str = "zero_shot") -> dict:
    """
    Generate a validated ScenarioConfig from a natural language description.

    Args:
        description: NL scenario description
        strategy: "zero_shot", "few_shot", or "cot"

    Returns:
        dict with 'scenario', 'raw_output', 'strategy', 'input'
    """
    system = BASE_SYSTEM_PROMPT
    if strategy == "few_shot":
        system += FEW_SHOT_SUFFIX
    elif strategy == "cot":
        system += COT_SUFFIX

    llm = build_llm()
    messages = [SystemMessage(content=system), HumanMessage(content=description)]

    print(f"\n[Agent] Strategy: {strategy}")
    print(f"[Agent] Input: {description}")

    # Retry up to 3 times if JSON is invalid
    raw_output = ""
    scenario_config = None

    for attempt in range(3):
        response = _invoke_with_retry(llm, messages)
        raw_output = response.content
        print(f"[Agent] Attempt {attempt+1} — got response")

        data = _extract_json(raw_output)
        if not data:
            print(f"[Agent] No JSON found, retrying...")
            messages.append(response)
            messages.append(HumanMessage(
                content="Your response did not contain valid JSON. Return ONLY the JSON object, nothing else."
            ))
            continue

        # Apply constraint fixes
        fixed = _apply_constraints(data)

        # Validate with Pydantic
        scenario_config = _validate(fixed, description)
        if scenario_config:
            print(f"[Agent] ✅ Valid scenario generated")
            break
        else:
            messages.append(response)
            messages.append(HumanMessage(
                content="The JSON failed validation. Fix speed limits and required fields, then return ONLY the corrected JSON."
            ))

    return {
        "scenario": scenario_config.model_dump() if scenario_config else None,
        "raw_output": raw_output,
        "strategy": strategy,
        "input": description,
    }