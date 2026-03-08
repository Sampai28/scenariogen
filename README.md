# ScenarioGen — Agentic AV Scenario Generator

Convert natural language driving descriptions into validated simulation configs for autonomous vehicle safety testing.

**Built for**: AV safety validation pipelines (CARLA, custom simulators)  
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
BEV Visualization (Matplotlib PNG) + CARLA Export
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

# With strategy
python main.py --input "..." --strategy few_shot

# With CARLA export
python main.py --input "..." --carla

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

## Project Structure

```
scenariogen/
├── src/
│   ├── schema.py          # Pydantic models + validators
│   ├── tools.py           # LangChain tools (constraint checker, config generator)
│   ├── agent.py           # LLM agent + generation pipeline
│   ├── visualizer.py      # Matplotlib BEV renderer
│   └── carla_export.py    # CARLA script + JSON exporter
├── examples/              # Pre-generated scenario JSONs + BEV PNGs
├── evals/
│   ├── ablation.py        # Ablation study runner
│   ├── ablation_results.json
│   └── ablation_table.md  # Results table
├── tests/
│   └── test_schema.py     # Pytest schema validation tests
├── app.py                 # Streamlit UI
├── main.py                # CLI entry point
└── requirements.txt
```

---

## Ablation Study

> Evaluated across 10 test scenarios (simple to complex) on schema validity, actor count accuracy, and plausibility (1-3).

<!-- PASTE ablation_table.md content here after running: python evals/ablation.py -->

---

## Example BEV Outputs

| Scenario | BEV |
|----------|-----|
| Pedestrian jaywalking at night in rain | ![](examples/placeholder.png) |

> See `/examples` folder for all pre-generated scenario JSONs and BEV PNGs.

---

## Resume Bullet Points

**ScenarioGen — Agentic AV Scenario Generator** | Python · LangChain · Pydantic · Groq · Matplotlib

- Built a LangChain agent pipeline converting natural language into validated AV simulation scenario configs, targeting autonomous vehicle safety testing workflows
- Designed Pydantic v2 schema enforcing physical plausibility constraints across 4 actor types, 4 weather conditions, and 4 road contexts with cross-field validators
- Ran ablation studies across zero-shot, few-shot, and chain-of-thought prompting strategies across 10 test scenarios, evaluating schema validity and plausibility
- Implemented CARLA-compatible export (Python script + JSON) and Matplotlib BEV visualization for direct simulator integration

---

## Interview Talking Points

**Problem**: AV safety validation requires thousands of diverse edge-case scenarios — writing them by hand doesn't scale.

**Approach**: Constraint checking and Pydantic validation as separate concerns after LLM generation — because a single LLM call that "usually" gets the schema right isn't good enough for safety-critical applications.

**Key challenge**: Getting a small local LLM to respect physical constraints (e.g. not generating a pedestrian moving at 30 m/s) — solved with a dedicated constraint checker that auto-corrects values and Pydantic as a hard rejection gate.

**How to scale**: Add a retrieval layer pulling from real-world incident logs to seed generation, and a feedback loop where simulator outcomes improve the few-shot examples over time.