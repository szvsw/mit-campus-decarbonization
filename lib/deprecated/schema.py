from typing import Optional, Annotated
import uuid
from pydantic import BaseModel, Field, field_validator, BeforeValidator, AfterValidator
import pandas as pd
import pandera as pa
from pandera.dtypes import DateTime
from pandera.typing import Index, DataFrame, Series
from pandera.engines.pandas_engine import PydanticModel


class Building(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    program_type: str
    footprint_area: float
    gfa: float
    n_floors: int
    height: float


class UBEMResultDataFrameSchema(pa.DataFrameModel):
    building_id: Index[str]  # TODO: should be uuid
    timestamp: Index[DateTime]
    Heating: float = pa.Field(ge=0)
    Cooling: float = pa.Field(ge=0)
    Lighting: float = pa.Field(ge=0)
    Equipment: float = pa.Field(ge=0)
    Fans: float = pa.Field(ge=0)
    Water: float = pa.Field(ge=0)
    Misc: float = pa.Field(ge=0)

    class Config:
        multiindex_name = "Building Energy Demand Row Index"
        multiindex_coerce = True
        multiindex_strict = True
        multiindex_ordered = True
        coerce = True
        strict = True
        ordered = True
        name = "UBEM Demand Results DataFrame"


class PlantFactorsDataFrameSchema(pa.DataFrameModel):
    timestep: Index[DateTime]
    carbon_factor: float = pa.Field(
        ge=0, description="Carbon factor in kgCO2e/kWh for each timestep"
    )
    cost_factor: float = pa.Field(
        ge=0, description="Cost factor in $/kWh for each timestep"
    )
    capacity: float = pa.Field(ge=0, description="Capacity in kW for each timestep")

    class Config:
        coerce = True
        strict = True
        ordered = True


class SupplyScenarioDataFrameSchema(PlantFactorsDataFrameSchema):
    plant_id: Index[str]

    class Config:
        multiindex_name = "Supply Scenario Row Index"
        multiindex_coerce = True
        multiindex_strict = True
        multiindex_ordered = True
        coerce = True
        strict = True
        ordered = True
        name = "Supply Scenario DataFrame"


class ConsumptionDataFrameSchema(SupplyScenarioDataFrameSchema):
    consumption: float = pa.Field(ge=0, description="Utilization of the plant in kW")

    class Config:
        multiindex_name = "Consumption Row Index"
        multiindex_coerce = True
        multiindex_strict = True
        multiindex_ordered = True
        coerce = True
        strict = True
        ordered = True
        name = "Consumption DataFrame"


class UBEMResult(BaseModel, arbitrary_types_allowed=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    df_url: str
    df: DataFrame[UBEMResultDataFrameSchema]


class Plant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    nominal_capacity: float = Field(
        ..., ge=0, description="Nominal capacity of the plant in kW"
    )
    embodied_carbon: Optional[float] = Field(
        None, ge=0, description="Embodied carbon in kgCO2e/kW"
    )


class PlantFactors(BaseModel, arbitrary_types_allowed=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plant_id: str = Field(..., description="ID of the plant")
    name: str
    description: str
    df: DataFrame[PlantFactorsDataFrameSchema]


class SupplyScenario(BaseModel, arbitrary_types_allowed=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    df_url: str
    df: DataFrame[SupplyScenarioDataFrameSchema]

    # TODO: should this instead take a list of plantfactors and then use a computed property?

    @field_validator("df", mode="before")
    def check_df(cls, v):
        if isinstance(v, pd.DataFrame):
            return v
        elif isinstance(v, list):
            if not all([isinstance(pf, PlantFactors) for pf in v]):
                raise ValueError("df must be a list of PlantFactors")
            dfs = [pf.df for pf in v]
            ids = [pf.plant_id for pf in v]
            # add id to index of each df as plant_id column
            updated_dfs = []
            for df, id in zip(dfs, ids):
                df["plant_id"] = id
                df = df.reset_index().set_index(["timestep", "plant_id"])
                updated_dfs.append(df)
            df = pd.concat(updated_dfs, axis=0)
            print(df)
            return df
        else:
            raise ValueError("df must be a list of PlantFactors or a DataFrame")


class PowerConsumptionScenario(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    df_url: str
    df: DataFrame[ConsumptionDataFrameSchema]


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "building_id": [str(uuid.uuid4()), str(uuid.uuid4())],
            "timestamp": ["2020-02-01 00:00:00", "2020-01-01 00:30:00"],
            "Heating": [1, 2],
            "Cooling": [1, 2],
            "Lighting": [1, 2],
            "Equipment": [1, 2],
            "Fans": [1, 2],
            "Water": [1, 2],
            "Misc": [1, 2],
        }
    ).set_index(["building_id", "timestamp"])

    ubem = UBEMResult(
        name="test",
        description="test",
        df_url="test",
        df=df,
    )

    nat_gas_plant = Plant(
        name="Natural Gas Plant",
        description="Natural gas plant",
        nominal_capacity=100,
        embodied_carbon=100,
    )

    u_reactor_plant = Plant(
        name="Micro Reactor",
        description="Micro reactor",
        nominal_capacity=100,
        embodied_carbon=100,
    )

    nat_gas_plant_factors = PlantFactors(
        plant_id=nat_gas_plant.id,
        name="Natural Gas Plant Capacity",
        description="Natural gas plant capacity",
        df_url="test",
        df=pd.DataFrame(
            {
                "timestep": ["2020-01-01 00:00:00", "2020-01-01 00:30:00"],
                "carbon_factor": [1.0, 2.0],
                "cost_factor": [1.0, 2.0],
                "capacity": [1.0, 2.0],
            }
        ).set_index("timestep"),
    )

    u_reactor_plant_factors = PlantFactors(
        plant_id=u_reactor_plant.id,
        name="Micro Reactor Capacity",
        description="Micro reactor capacity",
        df_url="test",
        df=pd.DataFrame(
            {
                "timestep": ["2020-02-01 00:00:00", "2020-01-01 00:30:00"],
                "carbon_factor": [1, 2],
                "cost_factor": [1, 2],
                "capacity": [1, 2],
            }
        ).set_index("timestep"),
    )

    supply_scenario = SupplyScenario(
        name="Supply Scenario",
        description="Supply Scenario",
        df_url="test",
        df=[nat_gas_plant_factors, u_reactor_plant_factors],
    )
