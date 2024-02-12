import math

import numpy as np
import pandas as pd
from sqlmodel import Session

from lib.models import (
    Building,
    BuildingSimulationResult,
    DemandScenario,
    DemandScenarioBuilding,
    DesignVector,
    PowerPlant,
    PowerPlantScenario,
    engine,
)

df = pd.read_csv("data/mit_buildings_info.csv")
results_df = pd.read_hdf("data/5out.hdf", key="results")
with Session(engine) as session:
    ds = DemandScenario(
        name="Test Data Scenario", description="Test Data Scenario", year_available=2030
    )
    dv = DesignVector(name="Test Design Vector")
    bsrs = []

    for i, row in df.iterrows():
        building = Building(
            name=row["BUILDING_NAME_LONG"],
            building_number=row["BUILDING_NUMBER"],
            group_level_1=row["BUILDING_GROUP_LEVEL1"],
            usage=row["CLUSTER_NUM"],
            gfa=(
                row["EXT_GROSS_AREA"] if not math.isnan(row["EXT_GROSS_AREA"]) else None
            ),
            height=(
                row["BUILDING_HEIGHT"]
                if not math.isnan(row["BUILDING_HEIGHT"])
                else None
            ),
            year=(
                row["YEAR_CONST_BEGAN"]
                if not math.isnan(row["YEAR_CONST_BEGAN"])
                else None
            ),
        )
        bsr = BuildingSimulationResult(
            demand_scenario_building=DemandScenarioBuilding(
                building=building, demand_scenario=ds, design_vector=dv
            ),
            heating=results_df[
                f"Zone Ideal Loads Supply Air Total Heating Energy"
            ].values.tolist(),
            cooling=results_df[
                f"Zone Ideal Loads Supply Air Total Cooling Energy"
            ].values.tolist(),
            lighting=results_df[f"Zone Lights Electricity Energy"].values.tolist(),
            equipment=results_df[
                f"Zone Electric Equipment Electricity Energy"
            ].values.tolist(),
        )
        bsrs.append(bsr)
    session.add_all(bsrs)

    pv = PowerPlant(
        name="Photovoltaics",
        description="Photovoltaic Panels on the roof",
        nominal_capacity=100,
        nominal_cost=0.01,
        nominal_emissions_factor=0,
    )
    grid = PowerPlant(
        name="Grid",
        description="The municipal power grid",
        nominal_capacity=99999,
        nominal_cost=0.24,
        nominal_emissions_factor=23,
    )
    cup = PowerPlant(
        name="CUP",
        description="The campus combined utility plant",
        nominal_capacity=40000,  # 40 MW
        nominal_cost=0.01,
        nominal_emissions_factor=0,
    )
    session.add_all([pv, grid, cup])

    t = np.linspace(0, 365 * 2 * np.pi, 8761)
    t = t[:-1]
    ts = pd.date_range(start="2020-01-01", end="2020-12-31", freq="h")[:-1]
    y = np.sin(t - 2 * np.pi / 3).clip(0, 1) * ((-np.cos(t / 365) + 1) / 4 + 0.5)

    pv_50 = PowerPlantScenario(
        power_plant=pv,
        name="50% PV Coverage",
        description="50% of the available roof space is covered in PV panels",
        year_available=2030,
        capacities=y * 500,
        emissions_factors=np.zeros(8760),
        cost_factors=np.zeros(8760),
    )
    pv_70 = PowerPlantScenario(
        power_plant=pv,
        name="75% PV Coverage",
        description="75% of the available roof space is covered in PV panels",
        year_available=2040,
        capacities=y * 75,
        emissions_factors=np.zeros(8760),
        cost_factors=np.zeros(8760),
    )
    session.add_all([pv_50, pv_70])
    session.commit()
