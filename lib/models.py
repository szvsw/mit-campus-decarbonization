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
        description (Optional[str]): The description of the demand scenario.
        year_available (Optional[int]): The year the demand scenario is available.
        buildings (list[Building]): The list of buildings associated with the demand scenario.
        demand_scenario_designs (list[DemandScenarioBuilding]): The list of demand scenario designs associated with the demand scenario.
    """

    __tablename__ = "DemandScenario"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(..., description="Name of the demand scenario")
    description: Optional[str] = Field(
        None, description="Description of the demand scenario"
    )
    year_available: Optional[int] = Field(
        ..., description="Year the demand scenario is available"
    )
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


class PowerPlant(SQLModel, table=True):
    """
    Represents a power plant in the decarbonization model.

    Attributes:
        id (Optional[int]): The ID of the power plant.
        name (str): The name of the power plant.
        description (Optional[str]): The description of the power plant.
        nominal_capacity (Optional[float]): The nominal capacity of the power plant in kW.
        nominal_cost (Optional[float]): The nominal cost of the power plant in $/kWh.
        nominal_emissions_factor (Optional[float]): The nominal emissions factor of the power plant in kgCO2/kWh.
        power_plant_scenarios (list[PowerPlantScenario]): The list of power plant scenarios associated with the power plant.
    """

    __tablename__ = "PowerPlant"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(..., description="Name of the power plant")
    description: Optional[str] = Field(
        None, description="Description of the power plant"
    )
    nominal_capacity: Optional[float] = Field(
        None, description="Nominal capacity of the power plant [kW]"
    )
    nominal_cost: Optional[float] = Field(
        None, description="Nominal cost of the power plant [$/kWh]"
    )
    nominal_emissions_factor: Optional[float] = Field(
        None, description="Nominal emissions factor of the power plant [kgCO2/kWh]"
    )
    power_plant_scenarios: list["PowerPlantScenario"] = Relationship(
        back_populates="power_plant"
    )


class PowerPlantScenario(SQLModel, table=True):
    """
    Represents a power plant scenario in the decarbonization model.

    Attributes:
        id (Optional[int]): The ID of the power plant scenario.
        power_plant_id (int): The ID of the associated power plant.
        power_plant (PowerPlant): The associated power plant.
        name (str): The name of the power plant scenario.
        description (Optional[str]): The description of the power plant scenario.
        year_available (Optional[int]): The year the power plant scenario is available.
        emissions_factors (Optional[np.ndarray]): The emissions factors for the power plant scenario.
        cost_factors (Optional[np.ndarray]): The cost factors for the power plant scenario.
        capacities (Optional[np.ndarray]): The capacities for the power plant scenario.
    """

    __tablename__ = "PowerPlantScenario"
    id: Optional[int] = Field(default=None, primary_key=True)
    power_plant_id: int = Field(..., foreign_key="PowerPlant.id")
    power_plant: PowerPlant = Relationship(back_populates="power_plant_scenarios")
    name: str = Field(..., description="Name of the power plant scenario")
    description: Optional[str] = Field(
        None, description="Description of the power plant scenario"
    )
    year_available: Optional[int] = Field(
        ..., description="Year the power plant scenario is available"
    )
    emissions_factors: np.ndarray = Field(None, sa_column=Column(Vector(8760)))
    cost_factors: np.ndarray = Field(None, sa_column=Column(Vector(8760)))
    capacities: np.ndarray = Field(None, sa_column=Column(Vector(8760)))

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "emissions_factors": self.emissions_factors,
                "cost_factors": self.cost_factors,
                "capacities": self.capacities,
            }
        )
        df.index = pd.date_range(start="1/1/2024", periods=8760, freq="h")
        df.index.name = "Timestamp"
        df = df.set_index(
            pd.Series([self.power_plant.id] * 8760, name="power_plant_id"), append=True
        )
        df = df.set_index(
            pd.Series([self.power_plant.name] * 8760, name="power_plant_name"),
            append=True,
        )
        df = df.set_index(
            pd.Series([self.name] * 8760, name="scenario_name"), append=True
        )
        df = df.set_index(pd.Series([self.id] * 8760, name="scenario_id"), append=True)
        return df

    class Config:
        arbitrary_types_allowed = True
