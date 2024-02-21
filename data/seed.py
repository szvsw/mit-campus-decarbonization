import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import Session, select
from tqdm.auto import tqdm

from lib.models import (
    Building,
    BuildingRetrofitLevelTypeEnum,
    BuildingSchedulesTypeEnum,
    BuildingSimulationResult,
    ClimateScenarioTypeEnum,
    DemandScenario,
    DemandScenarioBuilding,
    PowerPlant,
    PowerPlantScenario,
    engine,
)

if __name__ == "__main__":

    df = pd.read_csv("data/mit_buildings_info.csv")
    # results_df = pd.read_hdf("data/5out.hdf", key="results")
    pv_df = pd.read_hdf("data/pv.hdf", key="pv")
    with Session(engine) as session:
        """
        Create demand scenarios
        """
        demand_scenarios: list[DemandScenario] = []
        for year, cs, rs, ss in tqdm(
            list(
                product(
                    list(
                        range(2025, 2055, 5),
                    ),
                    ClimateScenarioTypeEnum,
                    BuildingRetrofitLevelTypeEnum,
                    BuildingSchedulesTypeEnum,
                )
            )
        ):
            ds = DemandScenario(
                name=f"year: {year}, climate: {cs.name.lower().replace('_', ' ')}, retrofit: {rs.name.lower()}, schedules: {ss.name.lower()}",
                year_available=year,
                climate_scenario=cs,
                building_retrofit_level=rs,
                building_schedules=ss,
            )
            session.add(ds)
            session.commit()
            session.refresh(ds)
            demand_scenarios.append(ds)

        bsrs = []

        """
        Load df of aggregated baseline results
        """
        with pd.HDFStore(Path("data") / "baseline_results_agg.hdf", "r") as f:
            keys = [k.lower()[1:-3] for k in f.keys()]
            raw_keys_lookup = {k.lower()[1:-3]: k for k in f.keys()}

        buildings = []
        for i, row in tqdm(list(df.iterrows()), desc="Creating Building..."):
            building = Building(
                name=row["BUILDING_NAME_LONG"],
                building_number=row["BUILDING_NUMBER"],
                group_level_1=row["BUILDING_GROUP_LEVEL1"],
                usage=row["CLUSTER_NUM"],
                gfa=(
                    row["EXT_GROSS_AREA"]
                    if not math.isnan(row["EXT_GROSS_AREA"])
                    else None
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
            session.add(building)
            session.commit()
            session.refresh(building)
            buildings.append(building)
            if (
                building.name.lower() in keys
                or building.building_number.lower() in keys
            ):
                key = (
                    building.name.lower()
                    if building.name.lower() in keys
                    else building.building_number.lower()
                )
                key = raw_keys_lookup[key]
                with pd.HDFStore(Path("data") / "baseline_results_agg.hdf", "r") as f:
                    results_df = f[key]

                for ds in tqdm(
                    demand_scenarios,
                    desc="Creating results for building in scenario...",
                ):
                    factor = 1
                    climate_scenario_factors_heating = {
                        ClimateScenarioTypeEnum.BUSINESS_AS_USUAL: 0.8,
                        ClimateScenarioTypeEnum.STABILIZED: 1,
                        ClimateScenarioTypeEnum.RUNAWAY: 0.6,
                        ClimateScenarioTypeEnum.IMPROVING: 1.1,
                    }
                    climate_scenario_factors_cooling = {
                        ClimateScenarioTypeEnum.BUSINESS_AS_USUAL: 1.2,
                        ClimateScenarioTypeEnum.STABILIZED: 1,
                        ClimateScenarioTypeEnum.RUNAWAY: 1.4,
                        ClimateScenarioTypeEnum.IMPROVING: 0.8,
                    }
                    schedules_factors = {
                        BuildingSchedulesTypeEnum.STANDARD: 1,
                        BuildingSchedulesTypeEnum.SETBACKS: 0.75,
                        BuildingSchedulesTypeEnum.ADVANCED: 0.5,
                    }
                    building_retrofit_factors = {
                        BuildingRetrofitLevelTypeEnum.BASELINE: 1,
                        BuildingRetrofitLevelTypeEnum.SHALLOW: 0.75,
                        BuildingRetrofitLevelTypeEnum.DEEP: 0.5,
                    }
                    chsf = climate_scenario_factors_heating[ds.climate_scenario]
                    ccsf = climate_scenario_factors_cooling[ds.climate_scenario]
                    sf = schedules_factors[ds.building_schedules]
                    rf = building_retrofit_factors[ds.building_retrofit_level]
                    yf = (ds.year_available - 2025) / 25
                    assert yf <= 1.0
                    h_rand = (yf * chsf + (1 - yf) * 1) * sf * rf
                    c_rand = (yf * ccsf + (1 - yf) * 1) * sf * rf 
                    e_rand = sf * rf
                    l_rand = sf * rf
                    # h_rand = (np.random.rand() * 0.1 + 0.95) * f
                    # c_rand = (np.random.rand() * 0.1 + 0.95) * f
                    # e_rand = (np.random.rand() * 0.1 + 0.95) * f
                    # l_rand = (np.random.rand() * 0.1 + 0.95) * f

                    bsr = BuildingSimulationResult(
                        demand_scenario_building=DemandScenarioBuilding(
                            building=building,
                            demand_scenario=ds,
                        ),
                        heating=(results_df.heating.values * h_rand).tolist(),
                        cooling=(results_df.cooling.values * c_rand).tolist(),
                        equipment=(results_df.equipment.values * e_rand).tolist(),
                        lighting=(results_df.lighting.values * l_rand).tolist(),
                    )
                    session.add(bsr)
                    session.commit()
            else:
                print(
                    f"Could not find energy results for {building.name} ({building.building_number})"
                )

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

        # t = np.linspace(0, 365 * 2 * np.pi, 8761)
        # t = t[:-1]
        # ts = pd.date_range(start="2020-01-01", end="2020-12-31", freq="h")[:-1]
        # y = np.sin(t - 2 * np.pi / 3).clip(0, 1) * ((-np.cos(t / 365) + 1) / 4 + 0.5)
        pv_75_cap = pv_df["kw 75 coverage"].values
        pv_50_cap = pv_df["kW 50 coverage"].values

        pv_50 = PowerPlantScenario(
            power_plant=pv,
            name="50% PV Coverage",
            description="50% of the available roof space is covered in PV panels",
            year_available=2030,
            capacities=pv_50_cap,
            emissions_factors=np.zeros(8760),
            cost_factors=np.zeros(8760),
        )
        pv_70 = PowerPlantScenario(
            power_plant=pv,
            name="75% PV Coverage",
            description="75% of the available roof space is covered in PV panels",
            year_available=2040,
            capacities=pv_75_cap,
            emissions_factors=np.zeros(8760),
            cost_factors=np.zeros(8760),
        )
        session.add_all([pv_50, pv_70])
        session.commit()
