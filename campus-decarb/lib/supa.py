import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from lib.client import client
from pydantic import BaseModel, Field, model_validator, validate_call


class BuildingBase(BaseModel, extra="forbid"):
    name: str = Field(..., description="Name of the building")
    gfa: float = Field(..., gt=100, description="Gross Floor Area of the building, m2")


class Building(BuildingBase):
    id: int = Field(..., description="Unique identifier for the building")
    created_at: str = Field(
        ..., description="Timestamp of the creation of the building"
    )

    @classmethod
    def get(cls, id: int):
        building = client.table("Building").select("*").eq("id", id).execute()
        if len(building.data) == 0:
            raise ValueError(f"Building with id {id} not found")

        return cls(**building.data[0])

    @classmethod
    @validate_call()
    def create(cls, building: BuildingBase):
        building = client.table("Building").insert(building.model_dump()).execute()
        return cls(**building.data[0])

    def commit(self):
        client.table("Building").upsert(self.model_dump()).execute()


class BuildingSimulationResult(BaseModel, arbitrary_types_allowed=True, extra="forbid"):
    id: int
    created_at: Optional[str] = Field(
        None, description="Timestamp of the creation of the record"
    )
    heating: np.ndarray
    cooling: np.ndarray
    lighting: np.ndarray
    equipment: np.ndarray
    pumps: np.ndarray
    fans: np.ndarray
    water: np.ndarray
    misc: np.ndarray

    # set up all np.ndarrays to serialize to list
    class Config:
        json_encoders = {np.ndarray: lambda v: v.tolist()}

    @property
    def building_id(self):
        res = (
            client.table("DemandScenarioBuilding")
            .select("building_id")
            .eq("id", self.id)
            .execute()
        )
        return res.data[0].get("building_id")

    @property
    def demand_scenario_id(self):
        res = (
            client.table("DemandScenarioBuilding")
            .select("demand_scenario_id")
            .eq("id", self.id)
            .execute()
        )
        return res.data[0].get("demand_scenario_id")

    @property
    def design_vector_id(self):
        res = (
            client.table("DemandScenarioBuilding")
            .select("design_vector_id")
            .eq("id", self.id)
            .execute()
        )
        return res.data[0].get("design_vector_id")

    @model_validator(mode="before")
    def cast_fields_to_numpy(cls, v):
        for key in v:
            if cls.model_fields[key].annotation == np.ndarray:
                if type(v[key]) == list:
                    v[key] = np.array(v[key])
                elif type(v[key]) == str:
                    v[key] = np.array(json.loads(v[key]))
                elif type(v[key]) == pd.Series:
                    v[key] = v[key].values
                else:
                    pass

                assert v[key].shape == (
                    8760,
                ), f"Field {key} must have shape (8760,) but has shape {v[key].shape}"
        return v

    @classmethod
    def get(cls, id: int):
        res = (
            client.table("BuildingSimulationResult").select("*").eq("id", id).execute()
        )
        if len(res.data) == 0:
            raise ValueError(f"BuildingSimulationResult with id {id} not found")

        return cls(**res.data[0])

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "heating": self.heating,
                "cooling": self.cooling,
                "lighting": self.lighting,
                "equipment": self.equipment,
                "pumps": self.pumps,
                "fans": self.fans,
                "water": self.water,
                "misc": self.misc,
            },
            index=pd.date_range(
                start="2024-01-01 00:00:00", periods=8760, freq="H", name="timestep"
            ),
        )
        df = df.set_index(
            pd.Series([self.id] * 8760, name="building_id"),
            append=True,
        )
        raise ValueError("Not finished implementing; decide on multiindex!")
        df = df.unstack(level="timestep")
        return df

    # def from_df(self, df: pd.DataFrame):
    #     series = df.loc[self.id]
    #     self.heating = series.heating.values
    #     self.cooling = series.cooling.values
    #     self.lighting = series.lighting.values
    #     self.equipment = series.equipment.values
    #     self.pumps = series.pumps.values
    #     self.fans = series.fans.values
    #     self.water = series.water.values
    #     self.misc = series.misc.values

    def commit(self):
        client.table("BuildingSimulationResult").upsert(
            self.model_dump(mode="json", exclude=["created_at"])
        ).execute()
