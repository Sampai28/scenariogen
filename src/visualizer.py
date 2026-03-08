"""
ScenarioGen — Bird's-Eye-View (BEV) visualization using Matplotlib.
Renders actors as colored arrows on a 2D road grid and saves as PNG.
"""

import math
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from src.schema import ScenarioConfig

# ── Color map per actor type ──────────────────────────────────────────────────

ACTOR_COLORS = {
    "vehicle":     "#2196F3",  # blue
    "pedestrian":  "#F44336",  # red
    "cyclist":     "#4CAF50",  # green
    "motorcycle":  "#FF9800",  # orange
}

EGO_COLOR = "#9C27B0"  # purple


# ── Main visualization function ───────────────────────────────────────────────

def visualize_scenario(config: ScenarioConfig, output_path: str = None) -> str:
    """
    Render a BEV visualization of a ScenarioConfig.

    Args:
        config: validated ScenarioConfig instance
        output_path: path to save PNG (default: examples/<scenario_id>.png)

    Returns:
        path to saved PNG file
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#1a1a2e")  # dark background = road surface
    fig.patch.set_facecolor("#0f0f1a")

    # ── Determine plot bounds from actor positions ────────────────────────────
    all_x = [a.start_position[0] for a in config.actors] + [0.0]
    all_y = [a.start_position[1] for a in config.actors] + [0.0]
    margin = 20
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    # Make it square
    span = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min, x_max = cx - span / 2, cx + span / 2
    y_min, y_max = cy - span / 2, cy + span / 2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Draw road grid ────────────────────────────────────────────────────────
    _draw_road(ax, config.environment.road_type, x_min, x_max, y_min, y_max)

    # ── Draw ego vehicle at origin ────────────────────────────────────────────
    _draw_actor(ax, x=0.0, y=0.0, heading=90.0, color=EGO_COLOR,
                label=f"EGO\n{config.ego_vehicle_speed_mps:.1f} m/s", size=2.0)

    # ── Draw each actor ───────────────────────────────────────────────────────
    for actor in config.actors:
        color = ACTOR_COLORS.get(actor.actor_type, "#FFFFFF")
        label = f"{actor.actor_type}\n{actor.speed_mps:.1f} m/s\n{actor.behavior}"
        _draw_actor(ax, x=actor.start_position[0], y=actor.start_position[1],
                    heading=actor.heading_degrees, color=color, label=label)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [mpatches.Patch(color=EGO_COLOR, label="Ego Vehicle")]
    seen_types = set()
    for actor in config.actors:
        if actor.actor_type not in seen_types:
            legend_handles.append(
                mpatches.Patch(
                    color=ACTOR_COLORS[actor.actor_type],
                    label=actor.actor_type.capitalize()
                )
            )
            seen_types.add(actor.actor_type)

    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

    # ── Title & labels ────────────────────────────────────────────────────────
    env = config.environment
    title = (f"Scenario {config.scenario_id}  |  "
             f"{env.road_type.replace('_',' ').title()}  |  "
             f"{env.weather.title()}  |  {env.time_of_day.title()}")
    ax.set_title(title, color="white", fontsize=11, pad=12)
    ax.set_xlabel("X (meters)", color="#aaa")
    ax.set_ylabel("Y (meters)", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── Description text ──────────────────────────────────────────────────────
    wrapped = _wrap_text(config.description, 80)
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom",
             color="#888", fontsize=8, style="italic")

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        os.makedirs("examples", exist_ok=True)
        output_path = f"examples/{config.scenario_id}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_actor(ax, x, y, heading, color, label, size=1.5):
    """Draw an actor as a colored arrow with a label."""
    rad = math.radians(heading)
    dx = math.cos(rad) * size * 2
    dy = math.sin(rad) * size * 2

    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.5, mutation_scale=20))

    # Circle at base
    circle = plt.Circle((x, y), size * 0.6, color=color, alpha=0.85, zorder=3)
    ax.add_patch(circle)

    # Label offset perpendicular to heading
    off_x = -math.sin(rad) * size * 3.5
    off_y =  math.cos(rad) * size * 3.5
    ax.text(x + off_x, y + off_y, label, color="white", fontsize=7,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#333", alpha=0.7, edgecolor="none"))


def _draw_road(ax, road_type, x_min, x_max, y_min, y_max):
    """Draw a simple road background based on road type."""
    road_color = "#2d2d2d"
    lane_color = "#555"
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    road_w = 12  # road width in meters

    if road_type == "intersection":
        # Horizontal road
        ax.add_patch(plt.Rectangle((x_min, cy - road_w), x_max - x_min,
                                    road_w * 2, color=road_color, zorder=0))
        # Vertical road
        ax.add_patch(plt.Rectangle((cx - road_w, y_min), road_w * 2,
                                    y_max - y_min, color=road_color, zorder=0))
        # Dashed center lines
        ax.axhline(cy, color=lane_color, linestyle="--", lw=1, alpha=0.6)
        ax.axvline(cx, color=lane_color, linestyle="--", lw=1, alpha=0.6)

    elif road_type == "highway":
        # Wide horizontal road with 3 lanes
        ax.add_patch(plt.Rectangle((x_min, cy - road_w * 1.5), x_max - x_min,
                                    road_w * 3, color=road_color, zorder=0))
        for offset in [-road_w * 0.5, road_w * 0.5]:
            ax.axhline(cy + offset, color=lane_color,
                       linestyle="--", lw=1, alpha=0.5)

    elif road_type == "residential":
        # Narrower road
        ax.add_patch(plt.Rectangle((x_min, cy - road_w * 0.7), x_max - x_min,
                                    road_w * 1.4, color=road_color, zorder=0))
        ax.axhline(cy, color=lane_color, linestyle="--", lw=1, alpha=0.4)

    elif road_type == "parking_lot":
        # Grid of parking spots
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min,
                                    y_max - y_min, color=road_color, zorder=0))
        for i in range(int(x_min), int(x_max), 6):
            ax.axvline(i, color=lane_color, lw=0.5, alpha=0.3)
        for j in range(int(y_min), int(y_max), 4):
            ax.axhline(j, color=lane_color, lw=0.5, alpha=0.3)


def _wrap_text(text, max_chars):
    """Simple word wrapper."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = f"{line} {w}".strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


# ── CLI helper ────────────────────────────────────────────────────────────────

def visualize_from_json(json_path: str) -> str:
    """Load a scenario JSON file and render its BEV visualization."""
    with open(json_path) as f:
        data = json.load(f)
    config = ScenarioConfig(**data)
    return visualize_scenario(config)