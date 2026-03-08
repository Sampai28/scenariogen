"""
Pydantic schema definitions for AV simulation scenarios.
All generated configs must pass validation before being used downstream.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal
import uuid


class Actor(BaseModel):
    actor_type: Literal["vehicle", "pedestrian", "cyclist", "motorcycle"]
    start_position: tuple[float, float]  # (x, y) in meters from origin
    heading_degrees: float               # 0 = East, 90 = North, 180 = West, 270 = South
    speed_mps: float                     # speed in meters per second
    behavior: str                        # e.g. "jaywalking", "braking", "turning left"

    @field_validator("speed_mps")
    @classmethod
    def speed_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("speed_mps must be non-negative")
        return v

    @field_validator("heading_degrees")
    @classmethod
    def heading_must_be_valid(cls, v):
        if not (0 <= v < 360):
            raise ValueError("heading_degrees must be between 0 and 359")
        return v

    @field_validator("behavior")
    @classmethod
    def behavior_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("behavior description cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def check_speed_by_actor_type(self):
        limits = {
            "vehicle": 50.0,
            "motorcycle": 50.0,
            "cyclist": 12.0,
            "pedestrian": 3.0,
        }
        limit = limits[self.actor_type]
        if self.speed_mps > limit:
            raise ValueError(
                self.actor_type + " speed " + str(self.speed_mps) +
                " m/s exceeds max allowed " + str(limit) + " m/s"
            )
        return self


class Environment(BaseModel):
    time_of_day: Literal["day", "dusk", "night"]
    weather: Literal["clear", "rain", "fog", "snow"]
    road_type: Literal["intersection", "highway", "residential", "parking_lot"]


class ScenarioConfig(BaseModel):
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    ego_vehicle_speed_mps: float = Field(ge=0, le=50)
    actors: List[Actor] = Field(min_length=1, max_length=6)
    environment: Environment
    duration_seconds: float = Field(gt=0, le=60)

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("description cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def check_highway_speed(self):
        if (
            self.environment.road_type == "highway"
            and self.ego_vehicle_speed_mps < 20
        ):
            raise ValueError(
                "Ego vehicle speed on highway should be at least 20 m/s, got " +
                str(self.ego_vehicle_speed_mps)
            )
        return self

    @model_validator(mode="after")
    def no_pedestrians_on_highway(self):
        if self.environment.road_type == "highway":
            for actor in self.actors:
                if actor.actor_type == "pedestrian":
                    raise ValueError("Pedestrians cannot appear in highway scenarios")
        return self