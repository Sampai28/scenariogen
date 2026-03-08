"""
ScenarioGen CLI — Generate AV simulation scenarios from natural language.

Usage:
    python main.py --input "a pedestrian jaywalks at night in heavy rain"
    python main.py --input "..." --strategy few_shot
    python main.py --input "..." --strategy cot --no-viz
"""

import argparse
import json
import os
from src.agent import generate_scenario
from src.schema import ScenarioConfig
from src.visualizer import visualize_scenario


def save_outputs(result: dict) -> tuple[str, str | None]:
    """Save scenario JSON and BEV PNG to examples/. Returns (json_path, png_path)."""
    os.makedirs("examples", exist_ok=True)

    scenario_data = result["scenario"]
    if not scenario_data:
        return None, None

    scenario_id = scenario_data.get("scenario_id", "unknown")
    json_path = f"examples/{scenario_id}.json"

    with open(json_path, "w") as f:
        json.dump(scenario_data, f, indent=2)

    return json_path, scenario_id


def main():
    parser = argparse.ArgumentParser(
        description="ScenarioGen — NL to AV Simulation Scenario Config"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help='Natural language scenario description (wrap in quotes)'
    )
    parser.add_argument(
        "--strategy", "-s", default="zero_shot",
        choices=["zero_shot", "few_shot", "cot"],
        help="Prompting strategy (default: zero_shot)"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip BEV visualization"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Custom output path for JSON (default: examples/<id>.json)"
    )
    parser.add_argument(
        "--carla", action="store_true",
        help="Also export a CARLA-compatible Python script and JSON"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ScenarioGen — AV Scenario Generator")
    print(f"{'='*60}")
    print(f"  Input    : {args.input}")
    print(f"  Strategy : {args.strategy}")
    print(f"{'='*60}\n")

    # Generate scenario
    result = generate_scenario(args.input, strategy=args.strategy)

    if not result["scenario"]:
        print("\n❌ Failed to generate a valid scenario.")
        print("Raw output:\n", result["raw_output"])
        return

    # Save JSON
    json_path, scenario_id = save_outputs(result)
    if args.output:
        json_path = args.output
        with open(json_path, "w") as f:
            json.dump(result["scenario"], f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Scenario generated successfully!")
    print(f"{'='*60}")
    print(f"\n📄 JSON saved to : {json_path}")

    # Pretty print the scenario
    print(f"\n{json.dumps(result['scenario'], indent=2)}")

    # BEV visualization
    if not args.no_viz:
        try:
            config = ScenarioConfig(**result["scenario"])
            png_path = f"examples/{scenario_id}.png"
            visualize_scenario(config, output_path=png_path)
            print(f"\n🖼️  BEV saved to   : {png_path}")
        except Exception as e:
            print(f"\n⚠️  Visualization failed: {e}")

    # CARLA export
    if args.carla:
        try:
            from src.carla_export import export_to_carla, export_to_carla_json
            config = ScenarioConfig(**result["scenario"])
            py_path   = export_to_carla(config)
            json_path2 = export_to_carla_json(config)
            print(f"\n🚗 CARLA script    : {py_path}")
            print(f"🚗 CARLA JSON      : {json_path2}")
        except Exception as e:
            print(f"\n⚠️  CARLA export failed: {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()