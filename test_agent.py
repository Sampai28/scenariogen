from src.agent import generate_scenario
from src.schema import ScenarioConfig
from src.visualizer import visualize_scenario

result = generate_scenario(
    "a pedestrian jaywalks at night during heavy rain while a cyclist cuts across an intersection",
    strategy="zero_shot"
)

print("\n=== FINAL OUTPUT ===")
print(result["raw_output"])

# Render BEV if we got a valid scenario
if result["scenario"]:
    config = ScenarioConfig(**result["scenario"])
    path = visualize_scenario(config)
    print(f"\n✅ BEV saved to: {path}")
else:
    print("\n⚠️ No valid scenario parsed from output")