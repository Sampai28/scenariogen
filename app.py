"""
ScenarioGen — Streamlit UI for interactive AV scenario generation.

Run with:
    streamlit run app.py
"""

import json
import os
import streamlit as st
from src.agent import generate_scenario
from src.schema import ScenarioConfig
from src.visualizer import visualize_scenario

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ScenarioGen",
    page_icon="🚗",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🚗 ScenarioGen")
st.markdown(
    "**Autonomous Vehicle Scenario Generator** — "
    "Type a driving scenario in plain English and get a validated simulation config."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    strategy = st.selectbox(
        "Prompting Strategy",
        options=["zero_shot", "few_shot", "cot"],
        format_func=lambda x: {
            "zero_shot": "Zero-Shot",
            "few_shot": "Few-Shot (3 examples)",
            "cot": "Chain-of-Thought",
        }[x],
        help="Controls how the LLM is prompted to generate the scenario."
    )

    st.markdown("---")
    st.markdown("**Strategy Guide**")
    st.markdown("- **Zero-Shot**: Fast, no examples")
    st.markdown("- **Few-Shot**: More accurate, uses examples")
    st.markdown("- **CoT**: Reasons step-by-step, most reliable")

    st.markdown("---")
    st.markdown("**Example Inputs**")
    examples = [
        "A pedestrian jaywalks at night during heavy rain while a cyclist cuts across an intersection",
        "Two vehicles collide at a foggy highway merge zone",
        "A child runs into the road from between parked cars in a residential area",
        "A motorcycle weaves between lanes on a highway at dusk",
        "A delivery van double-parks blocking a cyclist at an intersection in snow",
    ]
    selected_example = st.selectbox("Load an example", [""] + examples)

# ── Main input area ───────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    default_text = selected_example if selected_example else ""
    user_input = st.text_area(
        "Describe your driving scenario",
        value=default_text,
        height=100,
        placeholder="e.g. A pedestrian jaywalks at night during heavy rain while a cyclist cuts across an intersection",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Scenario", use_container_width=True, type="primary")
    mutate_btn = st.button("🔀 Generate Variant", use_container_width=True,
                           disabled="last_scenario" not in st.session_state)

# ── Generation ────────────────────────────────────────────────────────────────

if generate_btn and user_input.strip():
    with st.spinner("🤖 Agent is generating your scenario..."):
        result = generate_scenario(user_input.strip(), strategy=strategy)

    if result["scenario"]:
        st.session_state["last_scenario"] = result
        st.session_state["last_input"] = user_input.strip()
        st.rerun()
    else:
        st.error("❌ Failed to generate a valid scenario. Try rephrasing your input.")
        with st.expander("Raw agent output"):
            st.text(result["raw_output"])

elif generate_btn:
    st.warning("Please enter a scenario description.")

# ── Mutation ──────────────────────────────────────────────────────────────────

if mutate_btn and "last_scenario" in st.session_state:
    last = st.session_state["last_scenario"]
    mutation_prompt = (
        f"Generate an edge case variant of this scenario: {last['input']}. "
        "Change the weather, time of day, or add an additional actor to create "
        "a more challenging situation."
    )
    with st.spinner("🔀 Generating edge case variant..."):
        result = generate_scenario(mutation_prompt, strategy=strategy)

    if result["scenario"]:
        st.session_state["last_scenario"] = result
        st.session_state["last_input"] = mutation_prompt
        st.rerun()
    else:
        st.error("❌ Mutation failed. Try generating a base scenario first.")

# ── Results display ───────────────────────────────────────────────────────────

if "last_scenario" in st.session_state:
    result = st.session_state["last_scenario"]
    scenario_data = result["scenario"]

    st.divider()
    st.subheader("📊 Generated Scenario")

    # Top metrics row
    config = ScenarioConfig(**scenario_data)
    env = config.environment
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Scenario ID", config.scenario_id)
    m2.metric("Road Type", env.road_type.replace("_", " ").title())
    m3.metric("Weather", env.weather.title())
    m4.metric("Time", env.time_of_day.title())
    m5.metric("Actors", len(config.actors))

    st.markdown("---")

    # BEV + JSON side by side
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**🗺️ Bird's-Eye View**")
        os.makedirs("examples", exist_ok=True)
        png_path = f"examples/{config.scenario_id}_ui.png"
        visualize_scenario(config, output_path=png_path)
        st.image(png_path, use_container_width=True)

    with right:
        st.markdown("**📄 Scenario Config (JSON)**")
        st.json(scenario_data)

    st.markdown("---")

    # Actor breakdown table
    st.markdown("**🎭 Actor Breakdown**")
    actor_rows = []
    for a in config.actors:
        actor_rows.append({
            "Type": a.actor_type.title(),
            "Position (x, y)": f"({a.start_position[0]:.1f}, {a.start_position[1]:.1f})",
            "Heading (°)": f"{a.heading_degrees:.0f}°",
            "Speed (m/s)": f"{a.speed_mps:.1f}",
            "Behavior": a.behavior,
        })
    st.table(actor_rows)

    # Download buttons
    st.markdown("---")
    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(scenario_data, indent=2),
            file_name=f"scenario_{config.scenario_id}.json",
            mime="application/json",
        )
    with dl2:
        with open(png_path, "rb") as f:
            st.download_button(
                "⬇️ Download BEV PNG",
                data=f,
                file_name=f"scenario_{config.scenario_id}.png",
                mime="image/png",
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:12px'>"
    "ScenarioGen · Built with LangChain + Groq · For AV Safety Validation"
    "</div>",
    unsafe_allow_html=True,
)