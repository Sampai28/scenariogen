"""
Bird's-eye-view (BEV) visualization for AV scenarios using Matplotlib.
Renders actors as colored arrows on a 2D road grid and saves as PNG.
"""

import math
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.schema import ScenarioConfig

ACTOR_COLORS = {
    "vehicle":    "#2196F3",
    "pedestrian": "#F44336",
    "cyclist":    "#4CAF50",
    "motorcycle": "#FF9800",
}

EGO_COLOR = "#9C27B0"


def visualize_scenario(config, output_path=None):
    """
    Render a BEV visualization of a ScenarioConfig and save as PNG.

    Args:
        config: validated ScenarioConfig instance
        output_path: path to save PNG (default: examples/<scenario_id>.png)

    Returns:
        path to the saved PNG file
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    all_x = [a.start_position[0] for a in config.actors] + [0.0]
    all_y = [a.start_position[1] for a in config.actors] + [0.0]
    margin = 20
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    span = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min, x_max = cx - span / 2, cx + span / 2
    y_min, y_max = cy - span / 2, cy + span / 2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    draw_road(ax, config.environment.road_type, x_min, x_max, y_min, y_max)

    draw_actor(ax, x=0.0, y=0.0, heading=90.0, color=EGO_COLOR,
               label="EGO\n" + str(config.ego_vehicle_speed_mps) + " m/s", size=2.0)

    for actor in config.actors:
        color = ACTOR_COLORS.get(actor.actor_type, "#FFFFFF")
        label = actor.actor_type + "\n" + str(actor.speed_mps) + " m/s\n" + actor.behavior
        draw_actor(ax, x=actor.start_position[0], y=actor.start_position[1],
                   heading=actor.heading_degrees, color=color, label=label)

    legend_handles = [mpatches.Patch(color=EGO_COLOR, label="Ego Vehicle")]
    seen_types = set()
    for actor in config.actors:
        if actor.actor_type not in seen_types:
            legend_handles.append(
                mpatches.Patch(color=ACTOR_COLORS[actor.actor_type], label=actor.actor_type.capitalize())
            )
            seen_types.add(actor.actor_type)

    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

    env = config.environment
    title = (
        "Scenario " + config.scenario_id + "  |  " +
        env.road_type.replace("_", " ").title() + "  |  " +
        env.weather.title() + "  |  " +
        env.time_of_day.title()
    )
    ax.set_title(title, color="white", fontsize=11, pad=12)
    ax.set_xlabel("X (meters)", color="#aaa")
    ax.set_ylabel("Y (meters)", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    wrapped = wrap_text(config.description, 80)
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom",
             color="#888", fontsize=8, style="italic")

    if output_path is None:
        os.makedirs("examples", exist_ok=True)
        output_path = "examples/" + config.scenario_id + ".png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def draw_actor(ax, x, y, heading, color, label, size=1.5):
    rad = math.radians(heading)
    dx = math.cos(rad) * size * 2
    dy = math.sin(rad) * size * 2

    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=20))

    circle = plt.Circle((x, y), size * 0.6, color=color, alpha=0.85, zorder=3)
    ax.add_patch(circle)

    off_x = -math.sin(rad) * size * 3.5
    off_y =  math.cos(rad) * size * 3.5
    ax.text(x + off_x, y + off_y, label, color="white", fontsize=7,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#333", alpha=0.7, edgecolor="none"))


def draw_road(ax, road_type, x_min, x_max, y_min, y_max):
    road_color = "#2d2d2d"
    lane_color = "#555"
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    road_w = 12

    if road_type == "intersection":
        ax.add_patch(plt.Rectangle((x_min, cy - road_w), x_max - x_min, road_w * 2,
                                    color=road_color, zorder=0))
        ax.add_patch(plt.Rectangle((cx - road_w, y_min), road_w * 2, y_max - y_min,
                                    color=road_color, zorder=0))
        ax.axhline(cy, color=lane_color, linestyle="--", lw=1, alpha=0.6)
        ax.axvline(cx, color=lane_color, linestyle="--", lw=1, alpha=0.6)

    elif road_type == "highway":
        ax.add_patch(plt.Rectangle((x_min, cy - road_w * 1.5), x_max - x_min, road_w * 3,
                                    color=road_color, zorder=0))
        for offset in [-road_w * 0.5, road_w * 0.5]:
            ax.axhline(cy + offset, color=lane_color, linestyle="--", lw=1, alpha=0.5)

    elif road_type == "residential":
        ax.add_patch(plt.Rectangle((x_min, cy - road_w * 0.7), x_max - x_min, road_w * 1.4,
                                    color=road_color, zorder=0))
        ax.axhline(cy, color=lane_color, linestyle="--", lw=1, alpha=0.4)

    elif road_type == "parking_lot":
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    color=road_color, zorder=0))
        for i in range(int(x_min), int(x_max), 6):
            ax.axvline(i, color=lane_color, lw=0.5, alpha=0.3)
        for j in range(int(y_min), int(y_max), 4):
            ax.axhline(j, color=lane_color, lw=0.5, alpha=0.3)


def wrap_text(text, max_chars):
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


def visualize_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    config = ScenarioConfig(**data)
    return visualize_scenario(config)