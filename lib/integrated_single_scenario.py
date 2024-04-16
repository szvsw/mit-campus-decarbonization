import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from tqdm.autonotebook import tqdm

from lib import aws_settings as aws
from lib import config_settings as config

logger = logging.getLogger("INTEGRATED")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s:%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(config.log_level)

data_dir = config.data_dir / "iam"
UPGRADE_SEQUENCE_PATH = data_dir / "renovation_upgrade_sequence.csv"
DS_PHASED_SEQUENCE_PATH = data_dir / "district_phased.csv"


class ScenarioType(Enum):
    baseline = "baseline"
    partial = "partial"
    full = "full"


class GridScenarioType(Enum):
    bau = "bau"
    decarbonization = "decarbonization"
    cheap_ng = "cheap_ng"


class ClimateScenarioType(Enum):
    rcp45 = "rcp45"
    rcp85 = "rcp85"


class RenovationRateScenarioType(Enum):
    slow = "slow"
    fast = "fast"


class BaseScenario(BaseModel):

    climate: ClimateScenarioType
    grid: GridScenarioType
    retrofit: ScenarioType
    schedules: ScenarioType
    lab: ScenarioType
    district_system: ScenarioType
    nuclear: ScenarioType
    deep_geothermal: ScenarioType
    renovation_rate: RenovationRateScenarioType
    storage: ScenarioType
    carbon_capture: ScenarioType

    def __repr__(self) -> str:
        base = "\nScenario:"
        for key, val in self.model_dump().items():
            base += f"\n\t{key}={val.name}"
        return base

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def make_df(cls):
        opts = {}
        for key, t in cls.model_fields.items():
            en: Type[Enum] = t.annotation
            if not issubclass(en, Enum):
                raise ValueError(
                    f"{key} is not an Enum. Non enum keys should be defined in a child class."
                )
            opts[key] = en._member_names_
        ix = pd.MultiIndex.from_product(opts.values(), names=opts.keys())
        df = pd.DataFrame(index=ix, columns=["Scenario"])

        def create_scenario(row):
            return BaseScenario(**row)

        df["Scenario"] = df.index.to_frame().apply(create_scenario, axis=1)
        return df


class Scenario(BaseModel, arbitrary_types_allowed=True):
    artifacts_dir: DirectoryPath = Field(
        ..., description="Where to download artifacts to."
    )
    x: BaseScenario = Field(..., description="The base scenario.")

    temp_dir: tempfile.TemporaryDirectory = Field(
        default_factory=tempfile.TemporaryDirectory,
        description="Where to write artifacts for this scenario.",
    )
    profiles_path: str = Field(
        ..., description="The folder in the bucket where the buildings are stored."
    )
    source_df: Optional[pd.DataFrame] = Field(
        default=None, description="The source dataframe."
    )
    target_df: Optional[pd.DataFrame] = Field(
        default=None, description="The target dataframe."
    )
    reno_seq: pd.DataFrame = pd.read_csv(UPGRADE_SEQUENCE_PATH).set_index("building_id")
    scheduled_loads: Optional[pd.DataFrame] = Field(
        default=None,
        description="The scheduled loads after upgrade sequencing applied.",
    )

    def prep_artifacts(self):
        bucket = aws.bucket.get_secret_value()
        s3 = aws.s3_client

        # Listing files
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=self.profiles_path)
        logger.debug(f"Fetching files from {bucket}/{self.profiles_path}")
        all_files = []
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    all_files.append(obj["Key"])
        logger.debug(f"Found {len(all_files)} files in {self.profiles_path}")

        # Constructing the regex pattern based on the class attributes
        source_pattern = f".*EPW.{self.x.climate.value}.*_RETRO.baseline_SCHED.baseline_LAB.baseline.hdf"
        target_pattern = f".*EPW.{self.x.climate.value}.*_RETRO.{self.x.retrofit.value}_SCHED.{self.x.schedules.value if self.x.schedules != ScenarioType.partial else 'full'}_LAB.{self.x.lab.value}.hdf"
        # TODO: we also need to grab the retrofit/baseline/lab scenario for when schedules is partial
        if self.x.schedules == ScenarioType.partial:
            raise NotImplementedError("Partial schedules not implemented yet.")
        source_regex = re.compile(source_pattern)
        target_regex = re.compile(target_pattern)

        # Filtering files based on the constructed regex pattern
        source_files = [file for file in all_files if source_regex.search(file)]
        target_files = [file for file in all_files if target_regex.search(file)]
        logger.debug(
            f"Found {len(source_files)} source files and {len(target_files)} target files"
        )

        # Fetching and loading files
        logger.debug("Downloading artifacts...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            source_paths = list(
                tqdm(
                    executor.map(self.get_artifact, source_files),
                    total=len(source_files),
                    desc="Downloading source HDFs",
                )
            )
            target_paths = list(
                tqdm(
                    executor.map(self.get_artifact, target_files),
                    total=len(target_files),
                    desc="Downloading target HDFs",
                )
            )
            source_dfs = list(
                tqdm(
                    executor.map(self.open_hdf, source_paths),
                    total=len(source_paths),
                    desc="Opening source HDFs",
                )
            )
            target_dfs = list(
                tqdm(
                    executor.map(self.open_hdf, target_paths),
                    total=len(target_paths),
                    desc="Opening target HDFs",
                )
            )
            logger.debug("Finished downloading and opening artifacts.")
        logger.debug("Concatenating artifacts...")
        source = pd.concat(source_dfs).sort_index(level="sort_id")
        target = pd.concat(target_dfs).sort_index(level="sort_id")
        self.source_df = source
        self.target_df = target
        assert self.source_df.index.to_frame(index=False)[
            ["epw_scenario", "epw_year", "building_id"]
        ].equals(
            self.target_df.index.to_frame(index=False)[
                ["epw_scenario", "epw_year", "building_id"]
            ]
        )
        logger.info("Finished loading artifacts.")

    def get_artifact(self, artifact: str) -> Path:
        bucket = aws.bucket.get_secret_value()
        s3 = aws.s3_client
        artifact_fname = Path(artifact).name
        artifact_path = self.artifacts_dir / artifact_fname
        if artifact_path.exists():
            return artifact_path
        else:
            s3.download_file(bucket, artifact, artifact_path)
            return artifact_path

    def open_hdf(self, path: Path) -> pd.DataFrame:
        return pd.read_hdf(path)

    def bldg_loads(self, year: int, start_year=2025, final_year=2050):
        phase = year % 10
        interp = phase / 10
        lower_year = max(start_year, year - phase)
        upper_year = min(final_year, year - phase + 10)
        logger.debug(
            f"Interpolating between {lower_year} and {upper_year} for year {year}"
        )
        results = []
        for segment, df in [("source", self.source_df), ("target", self.target_df)]:
            df: pd.DataFrame = df
            lower = df.xs(lower_year, level="epw_year").reset_index(
                [
                    name
                    for name in df.index.names
                    if name not in ["building_id", "epw_year"]
                ],
                drop=True,
            )
            upper = df.xs(upper_year, level="epw_year").reset_index(
                [
                    name
                    for name in df.index.names
                    if name not in ["building_id", "epw_year"]
                ],
                drop=True,
            )
            data = lower * (1 - interp) + upper * interp
            results.append((segment, data))

        combined_results = pd.concat(
            {k: v for k, v in results},
            axis=0,
            keys=[k for k, v in results],
            names=["segment"],
        )
        combined_results = combined_results.set_index(
            pd.Index([year] * len(combined_results), name="epw_year"), append=True
        )
        return combined_results

    def make_load_sequence(self, start_year=2025, end_year=2050):
        logger.debug("Making load sequence...")
        dfs = []
        n_years = end_year - start_year + 1
        for i in range(n_years):
            year = start_year + i
            df = self.bldg_loads(year, start_year, end_year)
            dfs.append(df)
        df = pd.concat(dfs)
        upgrade_year = self.reno_seq.loc[
            df.index.get_level_values("building_id"),
            f"retrofit_year_{self.x.renovation_rate.name}",
        ].values
        partial_actually_ugprades = self.reno_seq.loc[
            df.index.get_level_values("building_id"), "sched_partial_flag"
        ].values
        is_upgraded = upgrade_year < df.index.get_level_values("epw_year")
        use_row = (is_upgraded & (df.index.get_level_values("segment") == "target")) | (
            ~is_upgraded & (df.index.get_level_values("segment") == "source")
        )
        scheduled = df[use_row]
        assert (
            len(scheduled)
            == len(self.bldg_loads(2025)) / len(["source", "target"]) * n_years
        )
        self.scheduled_loads = scheduled
        self.scheduled_loads.columns.names = ["end_use", "DateTime"]
        logger.info("Finished making load sequence.")

    def run_plants(self):
        """
        assign each year row as cup or ds
        """
        phases = pd.read_csv(DS_PHASED_SEQUENCE_PATH)
        scheduled = self.scheduled_loads
        if self.x.district_system == ScenarioType.baseline:
            # everything is cup
            assignments = pd.Series(index=scheduled.index, data="cup")
        elif self.x.district_system == ScenarioType.partial:
            raise NotImplementedError("Partial district system not implemented yet.")
            # everything goes to new district system
            # only some buildings are ds
        elif self.x.district_system == ScenarioType.full:
            raise NotImplementedError("Partial district system not implemented yet.")
            # some buildings are ds according to phasing
            # some buildings are cup
        if self.x.carbon_capture != ScenarioType.baseline:
            raise NotImplementedError("Only baseline carbon capture is supported")

        if self.x.storage != ScenarioType.baseline:
            raise NotImplementedError("Only baseline storage is supported")

        if self.x.nuclear != ScenarioType.baseline:
            raise NotImplementedError("Only baseline nuclear is supported")

        if self.x.deep_geothermal != ScenarioType.baseline:
            raise NotImplementedError("Only baseline deep geothermal is supported")

        if self.x.lab != ScenarioType.baseline:
            raise NotImplementedError("Only baseline lab is supported")

        if self.x.district_system != ScenarioType.baseline:
            raise NotImplementedError("Only baseline district system is supported")

        assignments = pd.Series(
            index=self.scheduled_loads.index,
            data="cup",
            name="thermal_assignment",
        )

        np.random.seed(42)
        assignments = pd.Series(
            index=self.scheduled_loads.index,
            # data=np.random.choice(["cup", "district"], len(s.scheduled_loads), replace=True),
            data=np.random.choice(["cup"], len(self.scheduled_loads), replace=True),
            name="thermal_assignment",
        )

        df = self.scheduled_loads.set_index(assignments, append=True)
        electricity_loads: pd.Series = (
            self.scheduled_loads["Electricity"]
            .groupby("epw_year")
            .sum()
            .stack(future_stack=True)
        )
        # TODO: vehicles
        vehicle_loads = pd.Series(
            index=electricity_loads.index, data=0, name="vehicle_loads"
        )
        electricity_loads = electricity_loads + vehicle_loads
        thermal_loads = (
            df[["Heating", "Cooling", "Water"]]
            .groupby(["epw_year", "thermal_assignment"])
            .sum()
        )
        thermal_loads["Heating"] = thermal_loads["Heating"] + thermal_loads["Water"]
        thermal_loads = thermal_loads.drop(columns="Water")
        thermal_loads: pd.DataFrame = thermal_loads.stack(
            level="end_use", future_stack=True
        ).unstack(level="epw_year")

        deep_geothermal_thermal_capacity = pd.Series(
            index=pd.MultiIndex.from_frame(
                thermal_loads.T.reset_index()[["epw_year", "DateTime"]],
                names=["epw_year", "DateTime"],
            ),
            data=0,
            name="clean_thermal_supply",
        ).sort_index()

        thermal_loads = thermal_loads.swaplevel(i=0, j=1, axis=0)
        thermal_loads = thermal_loads.T
        cup_heating_delta = (
            thermal_loads["Heating"]["cup"] - deep_geothermal_thermal_capacity
        )
        cup_heating_remaining = cup_heating_delta.clip(0, None)
        deep_geothermal_remaining = cup_heating_delta.clip(None, 0).abs()
        district_heating_delta = (
            (thermal_loads["Heating"]["district"] - deep_geothermal_remaining)
            if "district" in thermal_loads["Heating"].columns
            else pd.Series(
                index=cup_heating_delta.index,
                data=0,
            )
        )
        district_heating_remaining = district_heating_delta.clip(0, None)
        thermal_loads.loc[cup_heating_remaining.index, ("Heating", "cup")] = (
            cup_heating_remaining
        )
        thermal_loads.loc[district_heating_remaining.index, ("Heating", "district")] = (
            district_heating_remaining
        )
        deep_geothermal_thermal_utilization = (
            deep_geothermal_thermal_capacity - deep_geothermal_remaining
        ).rename("Deep Geothermal Utilization")

        thermal_loads = thermal_loads.T
        thermal_loads = thermal_loads.swaplevel(i=1, j=0, axis=0)

        MWh_per_mmBtu = 1 / 3.412  # MWh/mmBtu
        lbs_per_kg = 2.20462
        # Data from 2019_MIT_OS_CUP's_energy_Demand 4.421.pdf slide 15
        # CHP Mode
        cup_power = 15.8  # MW
        cup_turbine_gas_consumption_for_1hr = 214.2  # mmBtu/hr
        cup_kg_CO2eq_per_MWh_elec = 273
        cup_kg_CO2eq_per_kWh_elec = cup_kg_CO2eq_per_MWh_elec / 1000
        cup_kg_CO2eq_per_MWh_gas = 180  # TODO: TACIT - check this!!
        cup_kg_CO2eq_per_kWh_gas = cup_kg_CO2eq_per_MWh_gas / 1000
        cup_energy_for_1hr = cup_power  # MWh/hr
        cup_turbine_gas_consumption_for_1hr = (
            cup_turbine_gas_consumption_for_1hr * MWh_per_mmBtu
        )  # MWh/hr
        cup_elec_cop = (
            cup_energy_for_1hr / cup_turbine_gas_consumption_for_1hr
        )  # unitless

        # # Heat Only Mode
        # cup_heating_only_cop = 0.4
        # cup_steam_psi = 200
        # cup_kips_steam_per_hour = 114  # kips/hr
        # cup_mmBtu_gas_per_hour = 153  # mmBtu/hr
        # cup_lbs_CO2_eq_per_kip_steam = 158  # lbs CO2eq/kip
        # cup_lbs_CO2_eq_per_hour = (
        #     cup_lbs_CO2_eq_per_kip_steam * cup_kips_steam_per_hour
        # )  # lbs CO2eq/hr
        # cup_kg_CO2eq_per_hour = cup_lbs_CO2_eq_per_hour / lbs_per_kg  # kg CO2eq/hr
        # cup_kg_CO2eq_per_kip_steam = (
        #     cup_kg_CO2eq_per_hour / cup_kips_steam_per_hour
        # )  # kg CO2eq/kip
        # cup_mmBtu_steam_per_hour = cup_mmBtu_gas_per_hour * cup_heating_only_cop
        # cup_mmBtu_steam_per_kip_steam = (
        #     cup_mmBtu_steam_per_hour / cup_kips_steam_per_hour
        # )  # mmBtu/kip

        cup_elec_capacity = 40000
        cup_heat_cop = 0.4
        cup_elec_chiller_cop = 4.0
        cup_gas_chiller_cop = 1.8  # https://www.johnsoncontrols.co.th/-/media/jci/be/united-states/hvac-equipment/chillers/files/be_yst_res_steam-turbine-chillers.pdf?la=th&hash=924082786A5C7C3CAE87E54D05841880556CA301?la=th&hash=924082786A5C7C3CAE87E54D05841880556CA301
        chiller_balance = 0.5  # 0 = elec chiller, 1 = steam driven chiller
        ds_heat_cop = 3.5 if self.x.district_system == ScenarioType.partial else 3.3
        ds_cool_cop = 3.5 if self.x.district_system == ScenarioType.partial else 3.3

        # TODO: should cup chiller have a max capacity
        # TODO: should cup chiller only be served by cup electricity?
        # TODO: how should we control the cup steam vs elec chiller balance?
        # TODO: should cop's depend on weather conditions etc
        transfer_matrix = pd.DataFrame(
            {
                ("Cooling", "Gas"): [
                    1 / cup_gas_chiller_cop * chiller_balance,
                    0,
                ],
                ("Heating", "Gas"): [
                    1 / cup_heat_cop,
                    0,
                ],
                ("Cooling", "Electricity"): [
                    1 / cup_elec_chiller_cop * (1 - chiller_balance),
                    1 / ds_cool_cop,
                ],
                ("Heating", "Electricity"): [
                    0,
                    1 / ds_heat_cop,
                ],
            },
            columns=pd.MultiIndex.from_tuples(
                [
                    ("Cooling", "Gas"),
                    ("Heating", "Gas"),
                    ("Cooling", "Electricity"),
                    ("Heating", "Electricity"),
                ],
                names=["end_use", "fuel"],
            ),
            index=pd.Series(
                [
                    "cup",
                    "district",
                ],
                name="thermal_assignment",
            ),
        )

        stacked_mat: pd.DataFrame = transfer_matrix.stack(
            level=["end_use"],
            future_stack=True,
        ).fillna(0)

        coeffs: pd.DataFrame = stacked_mat.loc[thermal_loads.index]
        fuel_demands_from_conditioning: pd.DataFrame = thermal_loads.T.dot(coeffs)
        fuel_demands_from_conditioning.index = (
            fuel_demands_from_conditioning.index.swaplevel(0, 1)
        )
        fuel_demands_from_conditioning: pd.DataFrame = (
            fuel_demands_from_conditioning.sort_index()
        )

        # TODO: add clean energy sources
        clean_electricity = pd.Series(
            index=fuel_demands_from_conditioning.index,
            data=0,
            name="Electricity",
        )
        full_elec_demand = (
            fuel_demands_from_conditioning["Electricity"] + electricity_loads
        )
        elec_demand_delta = full_elec_demand - clean_electricity
        surplus_clean = elec_demand_delta.clip(None, 0).abs()
        unmet_elec_demand = elec_demand_delta.clip(0, None)

        # # TODO: decide between purchasing vs generating based off of cost to operate cup per kWh, emissions per kWh, and grid equivalents
        # TODO: using energy storage etc
        cup_elec_demand = unmet_elec_demand.clip(0, cup_elec_capacity)
        grid_elec_demand = unmet_elec_demand - cup_elec_demand

        grid_data = self.generate_grid_data(
            time_index=self.source_df.xs(2025, level="epw_year")["Electricity"].columns,
            epw_years=pd.Series(
                grid_elec_demand.index.get_level_values("epw_year")
                .unique()
                .sort_values(),
                name="epw_year",
            ),
        )
        # Compute Grid Emissions/Cost
        # TODO: export discounts using surplus
        grid_cost_per_kWh = grid_data["energy_cost"] / 1000  # $/kWh
        grid_emissions_per_kWh = grid_data["emission_rate"] / 1000  # kg CO2eq/kWh
        grid_cost = grid_cost_per_kWh * grid_elec_demand
        grid_emissions = grid_emissions_per_kWh * grid_elec_demand

        # compute cup carbon
        # TODO: check coeffs
        # TODO: carbon capture
        cup_factor_mode = "b"
        if cup_factor_mode == "a":
            cup_gas_for_elec = cup_elec_demand / cup_elec_cop  # kWh gas burnt
            cup_kg_CO2eq_due_to_elec = (
                cup_elec_demand * cup_kg_CO2eq_per_kWh_elec
            )  # kg CO2eq
            cup_gas_coeff = cup_kg_CO2eq_due_to_elec / cup_gas_for_elec  # kg CO2eq/kWh
            total_cup_gas = (
                cup_elec_demand / cup_elec_cop + fuel_demands_from_conditioning["Gas"]
            )  # kWh
            cup_emissions = total_cup_gas * cup_gas_coeff
        elif cup_factor_mode == "b":
            cup_elec_emissions = cup_elec_demand * cup_kg_CO2eq_per_kWh_elec
            cup_gas_emissions = (
                fuel_demands_from_conditioning["Gas"] * cup_kg_CO2eq_per_kWh_gas
            )
            cup_emissions = cup_gas_emissions + cup_elec_emissions
        # TODO: cup costs

        total_emissions = grid_emissions + cup_emissions
        print(total_emissions.groupby("epw_year").sum())

        return total_emissions

    def generate_grid_data(self, time_index: pd.Index, epw_years: pd.Series):
        # Grid emissions setup
        grid_data = pd.read_csv(
            "data/iam/grid_data.csv", header=[0, 1, 2, 3, 4, 5], index_col=0
        )
        grid_data = grid_data.sort_index(axis=1).droplevel([0, 1, -1], axis=1)
        grid_data.index = time_index
        grid_data.columns.names = [
            lev if lev != "year" else "epw_year" for lev in (grid_data.columns.names)
        ]
        grid_data = grid_data.rename(
            columns={
                "1": "bau",
                "2": "cheap_ng",
                "3": "decarbonization",
            }
        )  # TODO: replace with enums
        grid_data = grid_data.rename(
            columns={
                yr: int(yr)
                for yr in grid_data.columns.get_level_values("epw_year").unique()
            }
        )
        grid_data = (
            grid_data.stack(level="epw_year", future_stack=True)
            .swaplevel(i=0, j=1, axis=0)
            .sort_index()
        )
        grid_data = grid_data.sort_index(axis=1)
        grid_data = grid_data[self.x.grid.name]

        # populate intermediate years
        grid_data = grid_data.unstack(level="DateTime")
        years = epw_years
        a: pd.DataFrame = grid_data.loc[years - years % 5]
        b: pd.DataFrame = grid_data.loc[
            ((years - years % 5 + 5)).clip(years.min(), years.max())
        ]
        alpha = 1 - ((years % 5) / 5)
        beta = 1 - alpha
        a.index = years
        b.index = years
        grid_data = a * alpha.values.reshape(-1, 1) + b * beta.values.reshape(-1, 1)
        grid_data = grid_data.stack(level="DateTime", future_stack=True)
        return grid_data

    def run(self):
        logger.info("----")
        logger.info(self.x)
        self.prep_artifacts()
        self.make_load_sequence()
        self.run_plants()
        logger.info("----")


if __name__ == "__main__":
    # scenarios_df = BaseScenario.make_df()
    # base_ix = 0
    # profiles_path = "sdl-epengine/dev/dde9a86e/results"

    # with tempfile.TemporaryDirectory() as artifacts_dir:

    #     base_scenario: BaseScenario = scenarios_df.iloc[base_ix].Scenario
    #     s = Scenario(
    #         artifacts_dir=artifacts_dir,
    #         profiles_path=profiles_path,
    #         x=base_scenario,
    #     )
    #     s.prep_artifacts()

    # assert not s.artifacts_dir.exists()
    # s.source_df
    pass
