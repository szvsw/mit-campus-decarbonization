from typing import Optional

import numpy as np
import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, Relationship, SQLModel, create_engine

from lib import supa_settings

engine = create_engine(supa_settings.connection_string, echo=False)


class DemandScenarioBuilding(SQLModel, table=True):
    """
    Represents a building in a demand scenario.

    Attributes:
        id (Optional[int]): The ID of the demand scenario building.
        demand_scenario_id (int): The ID of the associated demand scenario.
        demand_scenario (DemandScenario): The associated demand scenario.
        building_id (int): The ID of the associated building.
        building (Building): The associated building.
        design_vector_id (int): The ID of the associated design vector.
        design_vector (DesignVector): The associated design vector.
        simulation_result (BuildingSimulationResult): The simulation result for the building.
    """

    __tablename__ = "DemandScenarioBuilding"
    id: Optional[int] = Field(default=None, primary_key=True)
    demand_scenario_id: int = Field(..., foreign_key="DemandScenario.id")
    demand_scenario: "DemandScenario" = Relationship(
        back_populates="demand_scenario_designs"
    )
    building_id: int = Field(..., foreign_key="Building.id")
    building: "Building" = Relationship(back_populates="demand_scenario_designs")
    design_vector_id: int = Field(..., foreign_key="DesignVector.id")
    design_vector: "DesignVector" = Relationship(
        back_populates="demand_scenario_designs"
    )
    simulation_result: "BuildingSimulationResult" = Relationship(
        back_populates="demand_scenario_building"
    )


class Building(SQLModel, table=True):
    """
    Represents a building in the MIT Campus Decarbonization project.

    Attributes:
        id (Optional[int]): The ID of the building.
        name (str): The name of the building.
        gfa (Optional[float]): The Gross Floor Area of the building in square meters.
        building_number (Optional[str]): The MIT Building number.
        usage (Optional[str]): The usage of the building.
        group_level_1 (Optional[str]): The classification of the building (Non-CUP, CUP, etc).
        height (Optional[float]): The height of the building in meters.
        year (Optional[int]): The year of construction.
        demand_scenarios (list[DemandScenario]): The demand scenarios associated with the building.
        demand_scenario_designs (list[DemandScenarioBuilding]): The demand scenario designs associated with the building.
    """

    __tablename__ = "Building"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(..., description="Name of the building")
    gfa: Optional[float] = Field(
        ..., ge=0, description="Gross Floor Area of the building, m2"
    )
    building_number: Optional[str] = Field(None, description="MIT Building number")
    usage: Optional[str] = Field(None, description="Building usage")
    group_level_1: Optional[str] = Field(None, description="Non-CUP, CUP, etc")
    height: Optional[float] = Field(None, description="Height of the building, m")
    year: Optional[int] = Field(None, description="Year of construction")
    demand_scenarios: list["DemandScenario"] = Relationship(
        back_populates="buildings", link_model=DemandScenarioBuilding
    )
    demand_scenario_designs: list[DemandScenarioBuilding] = Relationship(
        back_populates="building"
    )


class DemandScenario(SQLModel, table=True):
    """
    Represents a demand scenario.

    Attributes:
        id (Optional[int]): The ID of the demand scenario.
        name (str): The name of the demand scenario.
        buildings (list[Building]): The list of buildings associated with the demand scenario.
        demand_scenario_designs (list[DemandScenarioBuilding]): The list of demand scenario designs associated with the demand scenario.
    """

    __tablename__ = "DemandScenario"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(..., description="Name of the demand scenario")
    buildings: list["Building"] = Relationship(
        back_populates="demand_scenarios", link_model=DemandScenarioBuilding
    )
    demand_scenario_designs: list[DemandScenarioBuilding] = Relationship(
        back_populates="demand_scenario"
    )


class DesignVector(SQLModel, table=True):
    """
    Represents a design vector for decarbonization of a campus building.

    Attributes:
        id (Optional[int]): The ID of the design vector.
        name (str): The name of the design vector.
        demand_scenario_designs (list[DemandScenarioBuilding]): The list of demand scenario designs associated with the design vector.
    """

    __tablename__ = "DesignVector"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(..., description="Name of the design vector")
    demand_scenario_designs: list[DemandScenarioBuilding] = Relationship(
        back_populates="design_vector"
    )


class BuildingSimulationResult(SQLModel, table=True):
    """
    Represents the simulation result for a building in the decarbonization model.

    Attributes:
        id (Optional[int]): The ID of the simulation result.
        demand_scenario_building (DemandScenarioBuilding): The demand scenario building associated with the simulation result.
        heating (Optional[np.ndarray]): The heating data for the building.
        cooling (Optional[np.ndarray]): The cooling data for the building.
        lighting (Optional[np.ndarray]): The lighting data for the building.
        equipment (Optional[np.ndarray]): The equipment data for the building.
    """

    __tablename__ = "BuildingSimulationResult"
    id: Optional[int] = Field(
        default=None, primary_key=True, foreign_key="DemandScenarioBuilding.id"
    )
    demand_scenario_building: DemandScenarioBuilding = Relationship(
        back_populates="simulation_result"
    )
    heating: Optional[np.ndarray] = Field(None, sa_column=Column(Vector(8760)))
    cooling: Optional[np.ndarray] = Field(None, sa_column=Column(Vector(8760)))
    lighting: Optional[np.ndarray] = Field(None, sa_column=Column(Vector(8760)))
    equipment: Optional[np.ndarray] = Field(None, sa_column=Column(Vector(8760)))

    def to_df(self) -> pd.DataFrame:
        """
        Convert the simulation result to a pandas DataFrame.

        Returns:
            pd.DataFrame: The simulation result as a DataFrame.
        """
        df = pd.DataFrame(
            {
                "heating": self.heating,
                "cooling": self.cooling,
                "lighting": self.lighting,
                "equipment": self.equipment,
            }
        )
        df.index = pd.date_range(start="1/1/2024", periods=8760, freq="h")
        df.index.name = "Timestamp"
        df = df.set_index(
            pd.Series(
                [self.demand_scenario_building.building.id] * 8760, name="building_id"
            ),
            append=True,
        )
        df = df.set_index(
            pd.Series(
                [self.demand_scenario_building.building.name] * 8760,
                name="building_name",
            ),
            append=True,
        )
        return df

    class Config:
        arbitrary_types_allowed = True
