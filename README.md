# ScenarioGen — Agentic AV Scenario Generator

Convert natural language driving descriptions into validated simulation configs for autonomous vehicle safety testing.

**Built for**: AV safety validation pipelines (CARLA, ROS 2 / Gazebo, custom simulators)  
**Stack**: Python · LangChain · Groq (Llama 3.1) · Pydantic v2 · Matplotlib · Streamlit

---

## Live Demo

```bash
python main.py --input "a pedestrian jaywalks at night during heavy rain while a cyclist cuts across an intersection"
```

Output:
```json
{
  "scenario_id": "a329b739",
  "description": "a pedestrian jaywalks at night...",
  "ego_vehicle_speed_mps": 15.0,
  "actors": [
    {"actor_type": "pedestrian", "speed_mps": 1.5, "behavior": "jaywalks"},
    {"actor_type": "cyclist",    "speed_mps": 5.0, "behavior": "cuts across intersection"}
  ],
  "environment": {"time_of_day": "night", "weather": "rain", "road_type": "intersection"},
  "duration_seconds": 10.0
}
```

---

## Architecture

```
User NL Input
      ↓
LangChain LLM (Groq / Llama 3.1-8b)
      ↓
  ┌─────────────────────────────────┐
  │  Step 1: JSON Generation        │  → LLM produces structured scenario
  │  Step 2: ConstraintChecker Tool │  → validates physical plausibility
  │  Step 3: Pydantic Validation    │  → hard schema enforcement
  └─────────────────────────────────┘
      ↓
Validated ScenarioConfig (JSON)
      ↓
BEV Visualization (Matplotlib PNG) + CARLA Export / ROS 2 Launch File
```

### Design Decisions

**Why tool decomposition + Pydantic instead of a single LLM call?**  
A single LLM call that "usually" returns valid JSON isn't good enough for safety-critical AV validation. We use a dedicated constraint checker to auto-correct physical implausibilities (e.g. a pedestrian at 30 m/s), then Pydantic as a hard gate that rejects anything still invalid. This mirrors production AV pipelines where data integrity is non-negotiable.

**Why Groq + Llama 3.1?**  
Zero cost, no credit card, runs entirely via API with sub-second latency. The free tier is sufficient for both development and demo purposes — no cloud deployment needed.

**Why Pydantic v2 for schema validation?**  
Pydantic's `field_validator` and `model_validator` let us encode domain knowledge (no pedestrians on highways, per-actor speed limits) as code rather than prompt instructions. Constraints enforced in code can't be hallucinated away.

---

## Features

- Natural language to structured JSON via LangChain + Groq LLM
- Physical plausibility validation — per-actor speed limits, road type rules
- Pydantic v2 schema with cross-field validators (highway speed, actor count)
- Bird's-eye-view visualization — actors as labeled arrows on a 2D road grid
- CARLA export — generates ready-to-run Python scripts + JSON configs
- ROS 2 export — generates Python launch files for Gazebo simulation
- Scenario mutator — generates edge case variants from existing scenarios
- 3 prompting strategies — zero-shot, few-shot, chain-of-thought
- Streamlit UI — interactive demo with download buttons
- Ablation eval table — quantitative comparison across strategies

---

## Setup

**Requirements**: Python 3.10+, free [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/YOUR_USERNAME/scenariogen
cd scenariogen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

---

## Usage

**CLI:**
```bash
# Basic
python main.py --input "a cyclist runs a red light at a foggy intersection"

# With prompting strategy
python main.py --input "..." --strategy few_shot

# Export to CARLA
python main.py --input "..." --carla

# Export to ROS 2 / Gazebo
python main.py --input "..." --ros

# Skip visualization
python main.py --input "..." --no-viz
```

**Streamlit UI:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run ablation eval:**
```bash
python evals/ablation.py
```

---

## Export Formats

### CARLA
Generates a ready-to-run Python script and JSON config for the CARLA simulator. Use `--carla` on the CLI or the CARLA export button in the Streamlit UI.

### ROS 2 / Gazebo
Generates a Python-based ROS 2 launch file (`scenario_launch.py`) that:
- Launches Gazebo with the appropriate world file for the scenario's road type
- Spawns the ego vehicle and all actors at their configured positions and headings
- Applies weather and lighting parameters via Gazebo plugins

Use `--ros` on the CLI or the ROS 2 export button in the Streamlit UI. The launch file is written to the current directory by default.

---

## Project Structure

```
scenariogen/
├── src/
│   ├── schema.py          # Pydantic models + validators
│   ├── tools.py           # LangChain tools (constraint checker, config generator)
│   ├── agent.py           # LLM agent + generation pipeline
│   ├── visualizer.py      # Matplotlib BEV renderer
│   ├── carla_export.py    # CARLA script + JSON exporter
│   └── ros_export.py      # ROS 2 / Gazebo launch file exporter
├── examples/              # Pre-generated scenario JSONs + BEV PNGs
├── evals/
│   ├── ablation.py        # Ablation study runner
│   ├── ablation_results.json
│   └── ablation_table.md  # Results table
├── tests/
│   ├── test_schema.py     # Pytest schema validation tests
│   └── test_ros_export.py # Pytest ROS 2 export tests
├── app.py                 # Streamlit UI
├── main.py                # CLI entry point
└── requirements.txt
```

---