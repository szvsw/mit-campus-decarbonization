import gc
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
GRID_EMISSIONS_PATH = data_dir / "grid_data.csv"
PV_AREA_PATH = data_dir / "rooftop_area.csv"
PV_RAD_PATH = data_dir / "rooftop_rad.csv"

HDF_CACHE = {}


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
    district: ScenarioType
    nuclear: ScenarioType
    deepgeo: ScenarioType
    renorate: RenovationRateScenarioType
    ess: ScenarioType
    ccs: ScenarioType
    pv: ScenarioType

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

    def to_index(self):
        values = tuple((v.name for v in self.model_dump().values()))
        names = list(self.model_dump().keys())
        index = pd.MultiIndex.from_tuples([values], names=names)
        return index


class Scenario(BaseModel, arbitrary_types_allowed=True):
    artifacts_dir: DirectoryPath = Field(
        ..., description="Where to download artifacts to."
    )
    x: BaseScenario = Field(..., description="The base scenario.")

    temp_dir: tempfile.TemporaryDirectory = Field(
        default_factory=tempfile.TemporaryDirectory,
        description="Where to write artifacts for this scenario.",
    )
    results_path: str = Field(
        ..., description="The folder in the bucket where the results are stored."
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
        source["Electricity"] = source["Electricity"]
        target["Electricity"] = target["Electricity"]
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
        if path.name in HDF_CACHE:
            return HDF_CACHE[path.name]
        df = pd.read_hdf(path)
        if aws.env != "prod":
            HDF_CACHE[path.name] = df
        return df

    def bldg_loads(self, year: int, start_year=2025, final_year=2050):
        phase = year % 10
        interp = phase / 10
        lower_year = max(start_year, year - phase)
        upper_year = min(final_year, year - phase + 10)
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
            f"retrofit_year_{self.x.renorate.name}",
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

        bldg_ids = self.source_df.index.get_level_values("building_id").unique()
        phases = self.generate_district_phases(bldg_ids)
        time_index = (
            self.scheduled_loads.groupby(level="epw_year")
            .first()
            .stack(level="DateTime", future_stack=True)
            .index
        )
        clean_capacity = self.generate_clean_capacity(ix=time_index)
        logger.warning("NO GRID ARBITRAGE SCENARIO")

        assignments = pd.Series(
            index=self.scheduled_loads.index,
            data="cup",
            name="thermal_assignment",
        )
        year_idx = self.scheduled_loads.index.get_level_values("epw_year")
        upgrade_years = phases.loc[
            self.scheduled_loads.index.get_level_values("building_id"),
            "district_upgrade_year",
        ]
        is_switched = year_idx >= upgrade_years
        # TODO: handle non cup bulidings according to phase?
        if self.x.district == ScenarioType.baseline:
            # everything is cup
            pass
        elif self.x.district == ScenarioType.partial:
            assignments[is_switched] = "district"
            # only some buildings are ds
        elif self.x.district == ScenarioType.full:
            # some buildings are ds according to phasing
            # some buildings are cup
            logger.warning("District FULL scenario Not Different Enough!")
            assignments[is_switched] = "district"

        logger.warning("ESS SCENARIOS NEEDS FINALIZING")
        if self.x.ess != ScenarioType.baseline:
            raise NotImplementedError("Only baseline ESS is supported")

        logger.warning("DEEP GEOTHERMAL SCENARIOS NEEDS FINALIZING")

        logger.warning("SCHEDULES PARTIAL needs finalizing")
        if self.x.schedules == ScenarioType.partial:
            raise NotImplementedError("Partial schedules not supported yet")

        assert (assignments.index == self.scheduled_loads.index).all()
        if aws.env == "prod":
            del self.source_df, self.target_df
            gc.collect()
        aggregate_building_profile = (
            self.scheduled_loads.groupby("epw_year")
            .sum()
            .stack(level="DateTime", future_stack=True)
        )
        df = self.scheduled_loads.set_index(assignments, append=True)
        electricity_loads: pd.Series = (
            self.scheduled_loads["Electricity"]
            .groupby("epw_year")
            .sum()
            .stack(future_stack=True)
        )

        logger.warning("VEHICLE LOADS NEEDED")
        vehicle_loads = pd.Series(
            index=electricity_loads.index, data=0, name="vehicle_loads"
        )
        electricity_loads = electricity_loads + vehicle_loads
        thermal_loads = (
            df[["Heating", "Cooling", "Water"]]
            .groupby(["epw_year", "thermal_assignment"])
            .sum()
        )
        if aws.env == "prod":
            del df, self.scheduled_loads
            gc.collect()
        full_ix = pd.MultiIndex.from_product(
            [
                thermal_loads.index.get_level_values("epw_year").unique(),
                ["cup", "district"],
            ],
            names=["epw_year", "thermal_assignment"],
        )
        mask = full_ix.isin(thermal_loads.index)
        missing_rows = full_ix[~mask]
        thermal_loads = (
            pd.concat(
                [
                    thermal_loads,
                    pd.DataFrame(
                        data=np.zeros((len(missing_rows), len(thermal_loads.columns))),
                        index=missing_rows,
                        columns=thermal_loads.columns,
                    ),
                ]
            )
            .fillna(0.0)
            .sort_index()
        )

        thermal_loads["Heating"] = thermal_loads["Heating"] + thermal_loads["Water"]

        thermal_loads = thermal_loads.drop(columns="Water")

        thermal_loads: pd.DataFrame = thermal_loads.stack(
            level="end_use", future_stack=True
        )
        thermal_loads = thermal_loads.unstack(level="epw_year")
        thermal_loads = (
            thermal_loads.swaplevel(0, 1, axis=1).T.sort_index().T
        )  # year then time

        if ("district", "Heating") not in thermal_loads.index:
            logger.debug("Adding empty district heating")
            thermal_loads.loc[("district", "Heating"), :] = 0.0
        if ("district", "Cooling") not in thermal_loads.index:
            logger.debug("Adding empty district cooling")
            thermal_loads.loc[("district", "Cooling"), :] = 0.0
        tups = [
            ("district", "Heating"),
            ("district", "Cooling"),
            ("cup", "Heating"),
            ("cup", "Cooling"),
        ]
        tests = [tup in thermal_loads.index for tup in tups]
        assert all(tests), f"Missing at least some thermal load tups: {tups}"
        thermal_loads = thermal_loads.T
        original_thermal_loads = thermal_loads.copy(deep=True)
        assert (thermal_loads.index == time_index).all()

        clean_thermal_capacity = clean_capacity.th.sum(axis=1).rename("clean_thermal")

        cup_heating_delta = thermal_loads["cup"]["Heating"] - clean_thermal_capacity
        cup_heating_remaining = cup_heating_delta.clip(0, None)
        clean_thermal_remaining = cup_heating_delta.clip(None, 0).abs()
        district_heating_delta = (
            (
                thermal_loads["district"]["Heating"]
                - clean_thermal_remaining
                * (
                    0.0 if self.x.district == ScenarioType.full else 1.0
                )  # deep geothermal can't provide heat to low temp system
            )
            if "district" in thermal_loads.columns
            else pd.Series(
                index=cup_heating_delta.index,
                data=0,
            )
        )
        district_heating_remaining = district_heating_delta.clip(0, None)
        clean_thermal_remaining = district_heating_delta.clip(None, 0).abs()
        thermal_loads.loc[cup_heating_remaining.index, ("cup", "Heating")] = (
            cup_heating_remaining
        )
        thermal_loads.loc[district_heating_remaining.index, ("district", "Heating")] = (
            district_heating_remaining
        )

        # TODO: should we try to knock off some steam-driven chiller energy?
        clean_thermal_utilization = (
            clean_thermal_capacity - clean_thermal_remaining
        ).rename("Clean Thermal Utilization")

        # thermal_loads = thermal_loads.T
        # thermal_loads = thermal_loads.swaplevel(i=1, j=0, axis=0)

        MWh_per_mmBtu = 1 / 3.412  # MWh/mmBtu
        lbs_per_kg = 2.20462
        steam_enthalpy_200psi_Btu_per_lb = 1199
        steam_enthalpy_200psi_mmBtu_per_kip = (
            steam_enthalpy_200psi_Btu_per_lb * 1000 / 1e6
        )
        cup_boiler_capacity_kps_per_hr = 80 * 2 + 100 + 60 + 100
        cup_hrsg_capacity_kps_per_hr_unfired = 80 * 2
        cup_hrsg_capacity_kps_per_hr = 180 * 2  # fired
        cup_boiler_capacity_mmBtu_per_hr = (
            cup_boiler_capacity_kps_per_hr * steam_enthalpy_200psi_mmBtu_per_kip
        )
        cup_hrsg_capacity_mmBtu_per_hr_unfired = (
            cup_hrsg_capacity_kps_per_hr_unfired * steam_enthalpy_200psi_mmBtu_per_kip
        )
        cup_hrsg_capacity_mmBtu_per_hr = (
            cup_hrsg_capacity_kps_per_hr * steam_enthalpy_200psi_mmBtu_per_kip
        )
        cup_boiler_capacity = cup_boiler_capacity_mmBtu_per_hr * MWh_per_mmBtu * 1000
        cup_hrsg_capacity_unfired = (
            cup_hrsg_capacity_mmBtu_per_hr_unfired * MWh_per_mmBtu * 1000
        )
        cup_hrsg_capacity = cup_hrsg_capacity_mmBtu_per_hr * MWh_per_mmBtu * 1000

        cup_elec_capacity = 36000
        cup_turbine_elec_cop = 0.29
        cup_turbine_heat_cop = 0.39
        # cup_hrsg_heat_cop = 0.39
        cup_hrsg_heat_cop = 0.89
        cup_boiler_heat_cop = 0.89
        logger.warning("CHANGE CHILLER COPs?")
        cup_elec_chiller_cop = 3.5
        cup_gas_chiller_cop = 1.2  # https://www.johnsoncontrols.co.th/-/media/jci/be/united-states/hvac-equipment/chillers/files/be_yst_res_steam-turbine-chillers.pdf?la=th&hash=924082786A5C7C3CAE87E54D05841880556CA301?la=th&hash=924082786A5C7C3CAE87E54D05841880556CA301
        chiller_balance = 0.5  # 0 = elec chiller, 1 = steam driven chiller
        grid_balance = 0.1  # percentage of base elec load to satisfy by grid.
        ds_heat_cop = 3.5 if self.x.district == ScenarioType.partial else 3.3
        ds_cool_cop = 3.5 if self.x.district == ScenarioType.partial else 3.3
        cup_carbon_coeff_kgCO2eq_per_kWh_gas = 0.181  # standard

        # reduce thermal heating demands by deep geothermal if available
        # split buildings between cup and district if available according to phasing
        # cup cooling -> split between chillers -> move chillers to gas and electricity using cops
        # district heating/cooling -> move to electricity using cops
        # reduce electricity by available clean electricity
        # convert reamining electricity to gas using turbine (up to capacity)
        # convert gas burned for elec to free heat using hrsg cop
        # reduce cup heating by free heat
        # split remaining heating between fired hrsg and boilers
        # burn gas at hrsg and boilers to supply remaining heat
        # convert gas to CO2 using coeffs.
        elec_chiller_load = thermal_loads["cup"]["Cooling"] * (1 - chiller_balance)
        gas_chiller_load = thermal_loads["cup"]["Cooling"] * chiller_balance
        elec_chiller_demand = elec_chiller_load / cup_elec_chiller_cop
        steam_chiller_demand = gas_chiller_load / cup_gas_chiller_cop
        elec_district_demand = (
            thermal_loads["district"]["Cooling"] / ds_cool_cop
            + thermal_loads["district"]["Heating"] / ds_heat_cop
        )
        elec_demand_from_thermal = elec_chiller_demand + elec_district_demand

        elec_demand = elec_demand_from_thermal + electricity_loads
        clean_electricity_capacity = clean_capacity.e.sum(axis=1).rename(
            "clean_electricity"
        )

        elec_demand_delta = elec_demand - clean_electricity_capacity
        surplus_clean = elec_demand_delta.clip(None, 0).abs()
        logger.warning("NOT USING SURPLUS CLEAN")
        unmet_elec_demand = elec_demand_delta.clip(0, None)
        base_grid_demand = unmet_elec_demand * grid_balance
        unmet_elec_demand = unmet_elec_demand - base_grid_demand
        cup_elec_demand = unmet_elec_demand.clip(0, cup_elec_capacity)
        grid_elec_import = unmet_elec_demand - cup_elec_demand + base_grid_demand
        gas_cup_turbine_demand = cup_elec_demand / cup_turbine_elec_cop
        free_cup_heating = (gas_cup_turbine_demand * cup_turbine_heat_cop).clip(
            0, cup_hrsg_capacity_unfired
        )
        steam_demand = thermal_loads["cup"]["Heating"] + steam_chiller_demand
        cup_heat_delta = steam_demand - free_cup_heating
        cup_heat_remaining = cup_heat_delta.clip(0, None)
        cup_fired_hrsg_load = cup_heat_remaining.clip(0, cup_hrsg_capacity)
        cup_boiler_load = cup_heat_remaining - cup_fired_hrsg_load
        assert (cup_boiler_load < cup_boiler_capacity).all()
        gas_boiler_demand = cup_boiler_load / cup_boiler_heat_cop
        gas_fired_hrsg_demand = cup_fired_hrsg_load / cup_hrsg_heat_cop
        total_gas_demand = (
            gas_boiler_demand
            + gas_fired_hrsg_demand
            + gas_cup_turbine_demand
            # + gas_chiller_demand
        )
        total_cup_emissions_before_capture = (
            total_gas_demand * cup_carbon_coeff_kgCO2eq_per_kWh_gas
        )
        logger.warning("Capture is not yet implemented!")
        capture_capacity = self.generate_carbon_capture_capacity(ix=time_index)
        total_cup_emissions_delta = (
            total_cup_emissions_before_capture - capture_capacity.pc
        )
        total_cup_emissions = total_cup_emissions_delta.clip(0, None)
        total_capture = total_cup_emissions_before_capture - total_cup_emissions
        kWh_per_tonne_capture = 100
        kWh_per_kg_capture = kWh_per_tonne_capture / 1000
        elec_due_to_capture = total_capture * kWh_per_kg_capture
        assert (
            elec_due_to_capture.max()
        ) < 2000, "Carbon capture is taking greater than 2 MW at a time step!"

        grid_data = self.generate_grid_data(ix=time_index)
        logger.warning("Should we switch to full grid when carbon coeff is lower?")
        # Compute Grid Emissions/Cost
        # TODO: export discounts using surplus
        grid_cost_per_kWh = grid_data["energy_cost"] / 1000  # $/kWh
        grid_emissions_per_kWh = grid_data["emission_rate"] / 1000  # kg CO2eq/kWh
        grid_emissions = grid_emissions_per_kWh * grid_elec_import
        grid_cost = grid_cost_per_kWh * grid_elec_import

        total_emissions_before_capture = grid_emissions + total_cup_emissions
        total_emissions_after_capture = (
            total_emissions_before_capture - capture_capacity.dac - capture_capacity.bc
        )
        emissions_df = pd.concat(
            [
                total_emissions_before_capture.rename(
                    "Total Emissions excluding DAC/BC"
                ),
                total_emissions_after_capture.rename("Total Emissions after DAC/BC"),
                grid_emissions.rename("Grid Emissions"),
                total_cup_emissions.rename("CUP Emissions After PC"),
            ],
            axis=1,
        )
        assert (emissions_df.index == time_index).all()
        electricity_df = pd.concat(
            [
                elec_demand,
                cup_elec_demand,
                grid_elec_import,
            ],
            axis=1,
            keys=["Final Demand", "CUP", "Grid"],
        )
        assert (electricity_df.index == time_index).all()
        # TODO: include renewable utilization etc

        cup_equipment_loads = pd.concat(
            [
                free_cup_heating.rename("free_heating"),
                cup_boiler_load.rename("boiler"),
                cup_fired_hrsg_load.rename("fired_hrsg"),
                elec_chiller_load.rename("elec_chiller"),
                gas_chiller_load.rename("gas_chiller"),
            ],
            axis=1,
            # keys=["Free Heating", "Boiler", "Fired HRGS", "Elec Chiller", "Gas Chiller"],
        )

        results_df = pd.concat(
            [
                emissions_df,
                # grid_data,
                electricity_df,
                clean_capacity.th,
                clean_capacity.e,
                original_thermal_loads["cup"],
                original_thermal_loads["district"],
                cup_equipment_loads,
                aggregate_building_profile,
            ],
            axis=1,
            keys=[
                "Emissions",
                # "Grid Factors",
                "Electricity",
                "Clean Thermal",
                "Clean Electricity",
                "CUP Thermal Assignment",
                "District Thermal Assignment",
                "CUP Thermal Utilization",
                "Aggregate Building Profile",
            ],
        )
        assert (results_df.index == time_index).all()
        # TODO: should cup chiller have a max capacity
        # TODO: how should we control the cup steam vs elec chiller balance?
        # TODO: should cop's depend on weather conditions etc
        # # TODO: decide between purchasing vs generating based off of cost to operate cup per kWh, emissions per kWh, and grid equivalents
        # TODO: using energy storage etc
        # Compute Grid Emissions/Cost
        # TODO: export discounts using surplus
        # compute cup carbon
        # TODO: carbon capture
        # TODO: cup costs
        return results_df

    def generate_grid_data(self, ix: pd.Index):
        logger.debug("Generating grid data...")
        # Grid emissions setup
        grid_data = pd.read_csv(
            GRID_EMISSIONS_PATH, header=[0, 1, 2, 3, 4, 5], index_col=0
        )
        grid_data = grid_data.sort_index(axis=1).droplevel([0, 1, -1], axis=1)

        dt = pd.Series(
            ix.get_level_values("DateTime").unique().sort_values(), name="DateTime"
        )
        grid_data.index = dt
        grid_data.columns.names = [
            lev if lev != "year" else "epw_year" for lev in (grid_data.columns.names)
        ]
        grid_data = grid_data.rename(
            columns={
                "1": GridScenarioType.bau.name,
                "2": GridScenarioType.cheap_ng.name,
                "3": GridScenarioType.decarbonization.name,
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

        years = pd.Series(
            ix.get_level_values("epw_year").unique().sort_values(), name="epw_year"
        )
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
        logger.debug("Done generating grid data.")
        return grid_data

    def generate_district_phases(
        self, bldg_ids, years_per_phase=3, duration_of_phase=4, base_year=2030
    ):
        logger.warning("NON-CUP BUILDINGS PLACED ON CUP ANYWAYS")
        logger.debug("Generating district phase assignments...")
        df = pd.read_csv(DS_PHASED_SEQUENCE_PATH)
        df = df[df.columns[:-2]]
        df = df.rename(columns={"Building Number": "building_id"}).set_index(
            "building_id"
        )
        phased_bldgs = df.index.get_level_values("building_id").unique()
        bldg_loads_bldgs = bldg_ids
        mask = bldg_loads_bldgs.str.upper().isin(phased_bldgs.str.upper())
        missing_bldgs = (
            bldg_loads_bldgs[~mask].get_level_values("building_id")
        ).rename("building_id")
        mask = phased_bldgs.str.upper().isin(bldg_loads_bldgs.str.upper())
        non_simualted_bldgs = phased_bldgs[~mask].get_level_values("building_id")
        df = df.drop(non_simualted_bldgs)
        # choices = np.random.choice(df.Phase.unique(), size=len(missing_bldgs), replace=True)
        choices = df.Phase.sample(n=len(missing_bldgs)).values
        df = pd.concat([df, pd.DataFrame(index=missing_bldgs)])
        df.loc[missing_bldgs, "CUP (Y/N)"] = "Y"
        df.loc[missing_bldgs, "Phase"] = choices
        df["CUP (Y/N)"] = (
            df["CUP (Y/N)"].apply(lambda x: True if "Y" else False).astype(bool)
        )

        df["district_upgrade_year"] = base_year + (
            (df.Phase * years_per_phase - years_per_phase)
            + np.random.randint(0, duration_of_phase, len(df))
        ).astype(int)
        assert df.Phase.isna().sum() == 0
        df = df.sort_index()
        bldg_ids = sorted(list(bldg_ids))
        df.index = pd.Index(bldg_ids, name="building_id")
        logger.debug("Done generating district phase assignments.")
        return df

    def generate_deep_geothermal_capacity(self, ix: pd.Index):
        zero_f = 0 if self.x.deepgeo == ScenarioType.baseline else 1
        logger.warning(
            "DEEP GEOTHERMAL THERMAAL CAPACITY NEEDS FINALIZING, including FACTORs"
        )
        # TODO: add capacity factors.
        deep_geothermal_thermal_capacity = (
            50000 if self.x.deepgeo == ScenarioType.partial else 0
        )
        deep_geothermal_e_capacity = 50000 if self.x.deepgeo == ScenarioType.full else 0
        dgt_start_year = 2040
        available = ix.get_level_values("epw_year") >= dgt_start_year
        dgt_heat = pd.Series(
            index=ix,
            data=0.0,
            name="th",
        ).sort_index()
        dgt_e = pd.Series(
            index=ix,
            data=0.0,
            name="e",
        ).sort_index()
        dgt_heat.loc[available] = deep_geothermal_thermal_capacity
        dgt_e.loc[available] = deep_geothermal_e_capacity
        return pd.concat([dgt_heat, dgt_e], axis=1) * zero_f

    def generate_carbon_capture_capacity(self, ix: pd.Index):
        logger.warning("CARBON CAPTURE AND STORAGE CAPACITY NEEDS FINALIZING")
        post_combustion_annual_capacity_tonnes = 150000.0
        direct_air_capture_annual_capacity_tonnes = 4000.0
        biological_capture_annual_capacity_tonnes = 5000.0
        post_combustion_capacity_hourly_kg = (
            post_combustion_annual_capacity_tonnes / 8760 * 1000
        )
        direct_air_capture_capacity_hourly_kg = (
            direct_air_capture_annual_capacity_tonnes / 8760 * 1000
        )
        biological_capture_capacity_hourly_kg = (
            biological_capture_annual_capacity_tonnes / 8760 * 1000
        )
        pc_ccs_start_year = 2035
        dac_ccs_start_year = 2035
        pc_available = ix.get_level_values("epw_year") >= pc_ccs_start_year
        dac_available = ix.get_level_values("epw_year") >= dac_ccs_start_year
        pc = pd.Series(
            index=ix,
            data=0.0,
            name="pc",
        ).sort_index()
        dac = pd.Series(
            index=ix,
            data=0.0,
            name="dac",
        ).sort_index()
        bc = pd.Series(
            index=ix,
            data=biological_capture_capacity_hourly_kg,
            name="bc",
        ).sort_index()

        if self.x.ccs != ScenarioType.baseline:
            pc.loc[pc_available] = post_combustion_capacity_hourly_kg
        if self.x.ccs == ScenarioType.full:
            dac.loc[dac_available] = direct_air_capture_capacity_hourly_kg
        return pd.concat([pc, dac, bc], axis=1)

    def generate_pv_capacity(self, ix: pd.Index):
        areas = pd.read_csv(PV_AREA_PATH)
        polygon_areas = areas["PV AREA"]
        building_areas = areas.groupby("BUILDING").sum()["PV AREA"]
        total_solar_irrad = pd.read_csv(PV_RAD_PATH)["Irradiance [W/m2]"]
        total_solar_irrad.index = ix.get_level_values("DateTime").unique().sort_values()
        efficiency = 0.15
        data = total_solar_irrad.loc[ix.get_level_values("DateTime")]
        pv_Wsqm = pd.Series(index=ix, data=data.values, name="pv") * efficiency
        peak_Wsqm = pv_Wsqm.max()
        peak_kWsqm = peak_Wsqm / 1000
        polygon_kW_peaks = polygon_areas * peak_kWsqm
        building_kW_peaks = building_areas * peak_kWsqm
        usable_polygons = polygon_kW_peaks >= 100
        usable_buildings = building_kW_peaks >= 100
        usable_areas_by_polygon = polygon_areas[usable_polygons].sum()
        usable_areas_by_building = building_areas[usable_buildings].sum()
        total_area = polygon_areas.sum()
        assert abs(total_area - building_areas.sum()) < 1e-1
        if self.x.pv == ScenarioType.baseline:
            total_pv_area = usable_areas_by_polygon
        elif self.x.pv == ScenarioType.partial:
            total_pv_area = usable_areas_by_building
        else:
            total_pv_area = total_area
        pv_W = pv_Wsqm * total_pv_area
        pv_kW = pv_W / 1000
        ramp_dur = 10  # years
        first_year = ix.get_level_values("epw_year").min() + 1
        final_year = first_year + ramp_dur
        start_val = 0.05  # initial install capacity
        final_val = 1.0  # final install capacity
        n_years_for_linspace = 1 + final_year - first_year
        interp_vals = (
            np.cos(np.linspace(np.pi, 2 * np.pi, n_years_for_linspace)) * 0.5 + 0.5
        )
        interp_range = final_val - start_val
        interp_vals = interp_vals * interp_range + start_val
        interp_years = np.arange(n_years_for_linspace) + first_year
        f = np.interp(ix.get_level_values("epw_year"), interp_years, interp_vals)
        f[ix.get_level_values("epw_year") < first_year] = start_val
        f[ix.get_level_values("epw_year") > final_year] = final_val
        pv = pv_kW * f
        return pv

    def generate_nuclear_capacity(self, ix: pd.Index):
        zero_f = 0 if self.x.nuclear == ScenarioType.baseline else 1
        # in partial scenario, we buy two units in 2030, come online in 2035
        # in full scenario, we buy five units in 2030, come online in 2035
        n_batteries = 2 if self.x.nuclear == ScenarioType.partial else 5
        battery_capacity_kWhe = 5000
        battery_capacity_kWhTh = 8000
        years = ix.get_level_values("epw_year")
        first_year = 2035
        mask = years < first_year
        nuclear_supply = pd.Series(index=ix, data=1, name="nuclear")
        nuclear_supply[mask] = 0.0
        f = (
            [0, 60, 70, 80, 85, 90]
            if self.x.nuclear == ScenarioType.partial
            else [0, 60, 72.5, 85, 90, 90]
        )

        f = np.interp(np.linspace(0, 5, 26), np.arange(6), f) / 100
        capacity_factors = pd.Series(
            f,
            index=pd.Index(range(2025, 2051, 1), name="epw_year"),
        )
        battery_capacities = []
        logger.warning(
            "NUCLEAR CAPACITY FACTOR: should implement longer outages rather than hourly"
        )
        for i in range(n_batteries):
            thresh = np.random.uniform(0, 1, len(nuclear_supply))
            comparator = capacity_factors.loc[
                nuclear_supply.index.get_level_values("epw_year")
            ]
            comparator.index = nuclear_supply.index
            battery_capacities.append(
                nuclear_supply * (thresh < comparator).astype(int)
            )
        nuclear_supply = pd.concat(battery_capacities, axis=1, keys=range(n_batteries))
        nuclear_supply = nuclear_supply.sum(axis=1).rename("nuclear_opcount")
        nuclear_e = (nuclear_supply * battery_capacity_kWhe).rename("e")
        nuclear_th = (nuclear_supply * battery_capacity_kWhTh).rename("th")
        return pd.concat([nuclear_e, nuclear_th], axis=1) * zero_f

    def generate_clean_capacity(self, ix: pd.Index):

        logger.debug("Generating clean capacity...")
        nuclear = self.generate_nuclear_capacity(ix)
        pv = self.generate_pv_capacity(ix)
        geothermal = self.generate_deep_geothermal_capacity(ix)
        clean_e = pd.concat(
            [nuclear.e, pv, geothermal.e], axis=1, keys=["nuclear", "pv", "deepgeo"]
        )
        clean_th = pd.concat(
            [nuclear.th, geothermal.th], axis=1, keys=["nuclear", "deepgeo"]
        )
        clean = pd.concat([clean_th, clean_e], axis=1, keys=["th", "e"])
        logger.debug("Done generating clean capacity.")

        return clean

    def generate_dfs_and_scheduled_loads(self, from_s3=False, to_s3=False):
        np.random.seed(32)
        assert not (from_s3 and to_s3)
        if from_s3:
            logger.debug("Loading dfs from s3 cache...")
            local_output_file = Path(self.artifacts_dir) / self.building_loads_fname
            if os.path.exists(local_output_file):
                logger.info(f"Already downloaded: {local_output_file}")
            else:
                logger.info(f"Downloading from s3: {local_output_file}")
                aws.s3_client.download_file(
                    Bucket=aws.bucket.get_secret_value(),
                    Key=f"{self.results_path}/cached_dfs/{self.building_loads_fname}",
                    Filename=local_output_file.as_posix(),
                )
            self.source_df = pd.read_hdf(local_output_file, key="source_df")
            self.target_df = pd.read_hdf(local_output_file, key="target_df")
            self.scheduled_loads = pd.read_hdf(local_output_file, key="scheduled_loads")
            logger.info("Done downloading dfs from cache.")
        else:
            self.prep_artifacts()
            self.make_load_sequence()

        if to_s3:
            logger.debug("Uploading dfs to cache...")
            local_output_dir = Path(self.temp_dir.name) / self.building_loads_fname
            self.source_df.to_hdf(local_output_dir, key="source_df", mode="w")
            self.target_df.to_hdf(local_output_dir, key="target_df", mode="a")
            self.scheduled_loads.to_hdf(
                local_output_dir, key="scheduled_loads", mode="a"
            )
            # upload to s3
            aws.s3_client.upload_file(
                Filename=local_output_dir.as_posix(),
                Bucket=aws.bucket.get_secret_value(),
                Key=f"{self.results_path}/cached_dfs/{self.building_loads_fname}",
            )
            logger.info("Done uploading dfs to cache.")

    def run(self, from_s3=False, to_s3=False):
        logger.info("----")
        logger.info(self.x)
        self.generate_dfs_and_scheduled_loads(from_s3=from_s3, to_s3=to_s3)
        if to_s3:
            return None
        np.random.seed(42)
        final = self.run_plants()
        logger.info("----")
        return final

    @property
    def building_loads_fname(self):
        fname = f"CLIMATE.{self.x.climate.name}_RETRO.{self.x.retrofit.name}_SCHED.{self.x.schedules.name}_LAB.{self.x.lab.name}_RENORATE.{self.x.renorate.name}.hdf"
        return fname

    @property
    def fname(self):
        formatted_name = (
            str(self.x)
            .replace("\t", "")
            .replace("\n", "_")
            .replace("Scenario:", "")
            .replace("=", ".")
        )
        fname = f"{formatted_name}.hdf"
        return fname

    @property
    def local_output_path(self):
        return Path(self.temp_dir.name) / self.fname


TEST_SCENARIO = BaseScenario(
    renorate="slow",
    climate="rcp45",
    grid="bau",
    retrofit="baseline",
    schedules="baseline",
    lab="baseline",
    district="baseline",
    nuclear="baseline",
    deepgeo="baseline",
    ess="baseline",
    ccs="baseline",
    pv="baseline",
)


if __name__ == "__main__":
    import argparse
    import shutil
    import tempfile

    batch = "cc9e9cff-64f0-4fbf-b17a-2f829837f896"
    batch = "fd346eec-ef8b-4a06-a006-5a219c9246c4"

    scenarios_df = BaseScenario.make_df()
    mask = (
        (scenarios_df.index.get_level_values("ess") == ScenarioType.baseline.name)
        & (scenarios_df.index.get_level_values("ccs") != ScenarioType.full.name)
        & (
            scenarios_df.index.get_level_values("schedules")
            != ScenarioType.partial.name
        )
    )
    scenarios_df = scenarios_df[mask]
    array_job_ix = os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", None)

    base_ix = int(array_job_ix) if array_job_ix is not None else None

    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--batch_id", type=str, default=batch)
    parser.add_argument("--use_test_scenario", type=bool, default=False)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--from_s3", type=bool, default=True)
    parser.add_argument("--to_s3", type=bool, default=False)
    args = parser.parse_args()
    stride = args.stride
    batch_id = args.batch_id
    use_test_scenario = args.use_test_scenario
    run_name = args.run_name
    offset = args.offset
    from_s3 = args.from_s3
    to_s3 = args.to_s3
    profiles_path = f"sdl-epengine/dev/{batch_id[:8]}/results"
    results_path = f"mit-campus-decarbonization/results/{run_name}/{batch_id[:8]}"

    if to_s3:
        for key in BaseScenario.model_fields:
            if key not in [
                "renorate",
                "climate",
                "retrofit",
                "schedules",
                "lab",
                "grid",
            ]:
                mask = (
                    scenarios_df.index.get_level_values(key)
                    == ScenarioType.baseline.name
                )
                scenarios_df = scenarios_df[mask]
            if key == "grid":
                mask = (
                    scenarios_df.index.get_level_values(key)
                    == GridScenarioType.bau.name
                )
                scenarios_df = scenarios_df[mask]

    if array_job_ix is None:
        logger.info("Running locally!")
    else:
        logger.info(f"Running on AWS Batch: JOBIX: {array_job_ix}")
    k = base_ix if base_ix is not None else 0
    k = k + offset
    cleaned_dfs = []
    with tempfile.TemporaryDirectory() as artifacts_dir:
        while k < len(scenarios_df):
            logger.info(f"-------ROW:{k}-------")
            try:
                base_scenario = (
                    scenarios_df.iloc[k].Scenario
                    if not use_test_scenario
                    else TEST_SCENARIO
                )
                s = Scenario(
                    artifacts_dir=artifacts_dir,
                    profiles_path=profiles_path,
                    results_path=results_path,
                    x=base_scenario,
                )
                output_path = s.local_output_path
                tl: pd.DataFrame = s.run(to_s3=to_s3, from_s3=from_s3)
                if to_s3:
                    if stride == 0:
                        break
                    else:
                        k += stride
                        continue
                output_path = s.local_output_path
                annual_tl = tl.groupby("epw_year").sum()
                annual_tl.to_hdf(output_path, key="tl", mode="w")
                bucket_key = f"{results_path}/scenarios/{s.fname}"
                aws.s3_client.upload_file(
                    Filename=output_path.as_posix(),
                    Bucket=aws.bucket.get_secret_value(),
                    Key=bucket_key,
                )

                ix = base_scenario.to_index().repeat(len(annual_tl))
                annual_tl: pd.DataFrame = annual_tl.Emissions.rename(
                    columns={"Total Emissions after DAC/BC": "Emissions"}
                )[["Emissions"]]
                annual_tl = annual_tl.set_index(ix, append=True)
                logger.warning("TODO: COST")
                annual_tl["Cost"] = annual_tl.Emissions * 0.0002
                logger.warning("TODO: RISK and INNOVATION")
                annual_tl["Risk"] = 10
                annual_tl["Innovation"] = 15
                # add a level to the multi index with all velues set to k
                anuual_tl = annual_tl.set_index(
                    pd.Index(
                        [k] * len(annual_tl),
                        name="scenario_id",
                    ),
                    append=True,
                )

                cleaned_dfs.append(annual_tl)
                s.temp_dir.cleanup()
                if aws.env == "prod":
                    del annual_tl, ix, s, tl
                    gc.collect()

            except Exception as e:
                logger.error(
                    f"The following scenario failed! {base_scenario}", exc_info=e
                )
            if stride == 0:
                break
            k += stride
        if to_s3:
            exit()
        logger.info("Concatenating results...")
        df = pd.concat(cleaned_dfs)
        local_output_path = Path(artifacts_dir) / "scenarios.hdf"
        df.to_hdf(local_output_path, key="tl", mode="w")
        logger.info("Uploading results...")
        base_ix = base_ix if base_ix is not None else 0
        offset = offset if offset is not None else 0
        aws.s3_client.upload_file(
            Filename=local_output_path.as_posix(),
            Bucket=aws.bucket.get_secret_value(),
            Key=f"{results_path}/WORKER-B{base_ix:04d}-O{offset:04d}-S{stride:04d}.hdf",
        )
        logger.info("done!")
