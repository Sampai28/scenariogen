"""
ScenarioGen — Ablation study comparing zero_shot, few_shot, and cot strategies.

Evaluates each strategy across 10 test inputs on:
  - Schema validity   (pass/fail)
  - Actor count match (did we get at least the right number of actors?)
  - Plausibility      (manual 1–3 rating, auto-estimated here)

Run with:
    python evals/ablation.py

Outputs:
  - evals/ablation_results.json  — raw results
  - evals/ablation_table.md      — markdown table for README
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import generate_scenario
from src.schema import ScenarioConfig
from pydantic import ValidationError

# ── 10 test inputs (simple → complex) ────────────────────────────────────────

TEST_INPUTS = [
    # Simple
    ("simple_1",   "A car runs a red light at an intersection"),
    ("simple_2",   "A pedestrian crosses the road at night"),
    ("simple_3",   "A cyclist rides through a stop sign"),
    # Medium
    ("medium_1",   "A vehicle makes a sudden lane change on a foggy highway"),
    ("medium_2",   "A pedestrian jaywalks at night during heavy rain while a cyclist cuts across an intersection"),
    ("medium_3",   "A delivery truck double-parks blocking a cyclist at a rainy intersection"),
    # Complex
    ("complex_1",  "Two vehicles merge simultaneously on a highway at dusk while a motorcycle weaves between them"),
    ("complex_2",  "A child runs into the road from between parked cars in a residential area while a cyclist approaches from behind"),
    ("complex_3",  "A pedestrian and cyclist simultaneously cross against the light at a snowy intersection while a vehicle skids"),
    ("complex_4",  "Multiple vehicles queue at a foggy highway on-ramp while a motorcycle attempts to filter through"),
]

STRATEGIES = ["zero_shot", "few_shot", "cot"]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def estimate_plausibility(config: ScenarioConfig, description: str) -> int:
    """
    Auto-estimate plausibility score 1–3:
      3 = all speeds realistic, actors match description
      2 = minor issues (speed slightly off, missing actor)
      1 = major issues (wrong road type, impossible speeds)
    """
    score = 3
    desc_lower = description.lower()

    # Check actor types mentioned in description are present
    type_keywords = {
        "pedestrian": ["pedestrian", "person", "child", "walker"],
        "cyclist":    ["cyclist", "bike", "bicycle"],
        "motorcycle": ["motorcycle", "motorbike"],
        "vehicle":    ["vehicle", "car", "truck", "van"],
    }
    present_types = {a.actor_type for a in config.actors}
    for atype, keywords in type_keywords.items():
        if any(k in desc_lower for k in keywords) and atype not in present_types:
            score -= 1
            break

    # Check weather match
    weather_keywords = {"rain": "rain", "fog": "fog", "snow": "snow"}
    for weather, keyword in weather_keywords.items():
        if keyword in desc_lower and config.environment.weather != weather:
            score -= 1
            break

    # Check time of day
    if "night" in desc_lower and config.environment.time_of_day != "night":
        score -= 1
    if "dusk" in desc_lower and config.environment.time_of_day != "dusk":
        score -= 0  # minor, don't penalize

    return max(1, score)


def count_expected_actors(description: str) -> int:
    """Rough estimate of expected actor count from description."""
    desc_lower = description.lower()
    count = 0
    keywords = ["pedestrian", "person", "child", "cyclist", "bike",
                "motorcycle", "vehicle", "car", "truck", "van"]
    for k in keywords:
        count += desc_lower.count(k)
    return max(1, min(count, 6))


def evaluate_result(result: dict, description: str) -> dict:
    """Evaluate a single generation result."""
    eval_row = {
        "schema_valid": False,
        "actor_count": 0,
        "expected_actors": count_expected_actors(description),
        "actor_count_match": False,
        "plausibility": 1,
        "error": None,
    }

    if not result["scenario"]:
        eval_row["error"] = "No scenario parsed from output"
        return eval_row

    try:
        config = ScenarioConfig(**result["scenario"])
        eval_row["schema_valid"] = True
        eval_row["actor_count"] = len(config.actors)
        eval_row["actor_count_match"] = len(config.actors) >= eval_row["expected_actors"]
        eval_row["plausibility"] = estimate_plausibility(config, description)
    except ValidationError as e:
        eval_row["error"] = str(e)

    return eval_row


# ── Main runner ───────────────────────────────────────────────────────────────

def run_ablation():
    os.makedirs("evals", exist_ok=True)
    all_results = []

    total = len(TEST_INPUTS) * len(STRATEGIES)
    done = 0

    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        for test_id, description in TEST_INPUTS:
            done += 1
            print(f"  [{done}/{total}] {test_id}: {description[:50]}...")

            try:
                result = generate_scenario(description, strategy=strategy)
                eval_data = evaluate_result(result, description)
            except Exception as e:
                eval_data = {
                    "schema_valid": False,
                    "actor_count": 0,
                    "expected_actors": count_expected_actors(description),
                    "actor_count_match": False,
                    "plausibility": 1,
                    "error": str(e),
                }

            row = {
                "test_id": test_id,
                "description": description,
                "strategy": strategy,
                **eval_data,
            }
            all_results.append(row)

            status = "✅" if eval_data["schema_valid"] else "❌"
            plaus  = "⭐" * eval_data["plausibility"]
            print(f"    {status} valid={eval_data['schema_valid']}  "
                  f"actors={eval_data['actor_count']}  plaus={plaus}")

            # Respect Groq rate limits (30 req/min free tier)
            time.sleep(2)

    # Save raw results
    results_path = "evals/ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Raw results saved to {results_path}")

    # Generate markdown table
    _write_markdown_table(all_results)


def _write_markdown_table(results: list):
    """Write a markdown ablation table grouped by test input."""

    # Aggregate stats per strategy
    stats = {}
    for s in STRATEGIES:
        rows = [r for r in results if r["strategy"] == s]
        valid = sum(1 for r in rows if r["schema_valid"])
        match = sum(1 for r in rows if r["actor_count_match"])
        plaus = sum(r["plausibility"] for r in rows) / len(rows) if rows else 0
        stats[s] = {"valid": valid, "match": match, "plaus": plaus, "total": len(rows)}

    lines = [
        "## 📊 Ablation Study — Prompting Strategy Comparison",
        "",
        f"Evaluated across **{len(TEST_INPUTS)} test scenarios** ranging from simple to complex.",
        "",
        "### Summary",
        "",
        "| Strategy | Schema Valid | Actor Count Match | Avg Plausibility (1–3) |",
        "|----------|-------------|-------------------|------------------------|",
    ]
    for s in STRATEGIES:
        st = stats[s]
        lines.append(
            f"| {s.replace('_', '-').title()} | "
            f"{st['valid']}/{st['total']} | "
            f"{st['match']}/{st['total']} | "
            f"{st['plaus']:.2f} |"
        )

    lines += [
        "",
        "### Per-Scenario Results",
        "",
        "| Test ID | Description | Zero-Shot ✓ | Few-Shot ✓ | CoT ✓ | ZS Plaus | FS Plaus | CoT Plaus |",
        "|---------|-------------|-------------|-----------|-------|----------|----------|-----------|",
    ]

    for test_id, description in TEST_INPUTS:
        row_data = {r["strategy"]: r for r in results if r["test_id"] == test_id}
        zs = row_data.get("zero_shot", {})
        fs = row_data.get("few_shot", {})
        ct = row_data.get("cot", {})

        def valid_icon(r): return "✅" if r.get("schema_valid") else "❌"
        def plaus_str(r): return "⭐" * r.get("plausibility", 0) if r.get("schema_valid") else "—"

        short_desc = description[:45] + "..." if len(description) > 45 else description
        lines.append(
            f"| {test_id} | {short_desc} | "
            f"{valid_icon(zs)} | {valid_icon(fs)} | {valid_icon(ct)} | "
            f"{plaus_str(zs)} | {plaus_str(fs)} | {plaus_str(ct)} |"
        )

    table_path = "evals/ablation_table.md"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Markdown table saved to {table_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    run_ablation()