import json

import numpy as np
import pandas as pd
import taichi as ti
from supabase.client import PostgrestAPIError
from tqdm.autonotebook import tqdm

from lib.client import client
from lib.models import (
    BuildingRetrofitLevelTypeEnum,
    BuildingSchedulesTypeEnum,
    ClimateScenarioTypeEnum,
)


def make_npy():
    all_data = []
    i = 0
    query = "heating, cooling, lighting, equipment, DemandScenarioBuilding ( building_id, DemandScenario (climate_scenario, year_available, building_schedules, building_retrofit_level, lab_strategy))"
    while True:
        print("fetching page", i)
        try:
            r = (
                client.table("BuildingSimulationResult")
                .select(query)
                .order("id")
                .limit(1000)
                .offset(i * 1000)
                .execute()
            )
        except PostgrestAPIError as e:
            if e.code != "57014":
                raise e
            else:
                print("Passing on 57014")
        else:
            all_data.extend(r.data)
            i = i + 1
            print(len(r.data))
            print(len({b["DemandScenarioBuilding"]["building_id"] for b in all_data}))
            if len(r.data) < 1000:
                break

    print("Checking results counts")
    ids = set()
    climate_scenarios = set()
    retrofit_scenarios = set()
    lab_scenarios = set()
    schedules_scenarios = set()
    year_scenarios = set()
    for building in all_data:
        ids.add(building["DemandScenarioBuilding"]["building_id"])
        climate_scenarios.add(
            building["DemandScenarioBuilding"]["DemandScenario"]["climate_scenario"]
        )
        retrofit_scenarios.add(
            building["DemandScenarioBuilding"]["DemandScenario"][
                "building_retrofit_level"
            ]
        )
        lab_scenarios.add(
            building["DemandScenarioBuilding"]["DemandScenario"]["lab_strategy"]
        )
        schedules_scenarios.add(
            building["DemandScenarioBuilding"]["DemandScenario"]["building_schedules"]
        )
        year_scenarios.add(
            building["DemandScenarioBuilding"]["DemandScenario"]["year_available"]
        )
    id_map = {building: i for i, building in enumerate(sorted(ids))}
    year_map = {year: i for i, year in enumerate(sorted(year_scenarios))}
    n_climate_scenarios = len(climate_scenarios)
    n_retrofit_scenarios = len(retrofit_scenarios)
    n_lab_scenarios = len(lab_scenarios)
    n_schedules_scenarios = len(schedules_scenarios)
    n_year_evals = len(year_scenarios)
    n_buildings = len(ids)
    print(n_buildings)
    assert n_buildings == 90
    assert n_climate_scenarios == 4
    assert n_retrofit_scenarios == 3
    assert n_lab_scenarios == 3
    assert n_schedules_scenarios == 3
    assert n_year_evals == 6

    print("n_climate_scenarios", n_climate_scenarios)
    print("n_year_evals", n_year_evals)
    print("n_retrofit_scenarios", n_retrofit_scenarios)
    print("n_lab_scenarios", n_lab_scenarios)
    print("n_schedules_scenarios", n_schedules_scenarios)
    print("n_buildings", n_buildings)

    print("parsing results")
    final_array = (
        np.empty(
            shape=(
                n_climate_scenarios,  # climate_scenarios
                n_year_evals,  # year_evals
                n_retrofit_scenarios,  # retrofits
                n_lab_scenarios,  # lab
                n_schedules_scenarios,  # schedules
                n_buildings,  # buildings
                4,  # end uses
                8760,  # hours
            ),
            dtype=np.float32,
        )
        / 0
    )
    print(len(final_array.flatten()) / 1000000)
    pbar = tqdm(all_data, total=len(all_data))
    for result in pbar:
        pbar.set_description(
            f'Updating {result["DemandScenarioBuilding"]["building_id"]}'
        )
        heating = json.loads(result["heating"])
        cooling = json.loads(result["cooling"])
        equipment = json.loads(result["equipment"])
        lighting = json.loads(result["lighting"])

        # get climate int
        climate_scenario = result["DemandScenarioBuilding"]["DemandScenario"][
            "climate_scenario"
        ]
        climate_scenario = ClimateScenarioTypeEnum[climate_scenario].value

        # get retrofit int
        building_retrofit_level = result["DemandScenarioBuilding"]["DemandScenario"][
            "building_retrofit_level"
        ]
        building_retrofit_level = BuildingRetrofitLevelTypeEnum[
            building_retrofit_level
        ].value

        # get lab int
        lab_strategy = result["DemandScenarioBuilding"]["DemandScenario"][
            "lab_strategy"
        ]
        lab_strategy = BuildingRetrofitLevelTypeEnum[lab_strategy].value

        # get schedules int
        building_schedules = result["DemandScenarioBuilding"]["DemandScenario"][
            "building_schedules"
        ]
        building_schedules = BuildingSchedulesTypeEnum[building_schedules].value

        building_id = id_map[result["DemandScenarioBuilding"]["building_id"]]
        year = result["DemandScenarioBuilding"]["DemandScenario"]["year_available"]
        year = year_map[year]
        final_array[
            climate_scenario,
            year,
            building_retrofit_level,
            lab_strategy,
            building_schedules,
            building_id,
            0,
            :,
        ] = heating
        final_array[
            climate_scenario,
            year,
            building_retrofit_level,
            lab_strategy,
            building_schedules,
            building_id,
            1,
            :,
        ] = cooling
        final_array[
            climate_scenario,
            year,
            building_retrofit_level,
            lab_strategy,
            building_schedules,
            building_id,
            2,
            :,
        ] = lighting
        final_array[
            climate_scenario,
            year,
            building_retrofit_level,
            lab_strategy,
            building_schedules,
            building_id,
            3,
            :,
        ] = equipment
    assert np.isnan(final_array).sum() == 0
    np.save("data/all_scenarios.npy", final_array)


@ti.data_oriented
class DemandScenarioProjector:
    def __init__(
        self,
        *,
        n_simulation_passes: int,
        n_weather_scenarios: int,
        n_evals_per_building_scenario: int,
        n_years_per_eval: int,
        n_building_scenarios: int,
        n_lab_scenarios: int,
        n_schedules_scenarios: int,
        n_buildings: int,
        n_enduses: int,
        n_timesteps: int,
    ):
        self.n_simulation_passes = n_simulation_passes
        self.n_years_per_eval = n_years_per_eval
        n_years_per_sim = (n_evals_per_building_scenario - 1) * n_years_per_eval + 1
        self.n_years_per_sim = n_years_per_sim
        self.n_weather_scenarios = n_weather_scenarios
        self.n_evals_per_building_scenario = n_evals_per_building_scenario
        self.n_building_scenarios = n_building_scenarios
        self.n_lab_scenarios = n_lab_scenarios
        self.n_schedules_scenarios = n_schedules_scenarios
        self.n_buildings = n_buildings
        self.n_enduses = n_enduses
        self.n_timesteps = n_timesteps

        self.building_data = ti.field(
            dtype=ti.f32,
            shape=(
                n_weather_scenarios,
                n_evals_per_building_scenario,
                n_building_scenarios,
                n_lab_scenarios,
                n_schedules_scenarios,
                n_buildings,
                n_enduses,
                n_timesteps,
            ),
        )

        self.upgrade_years = ti.field(
            dtype=ti.i32, shape=(n_simulation_passes, n_buildings)
        )

        # TODO: could also store per weather scenario or retrofit rate
        self.results = ti.field(
            dtype=ti.f32,
            shape=(n_simulation_passes, n_years_per_sim, n_enduses, n_timesteps),
        )

    def load_from_np(self, data: np.ndarray):
        """
        Load the building data from a numpy array

        Args:
            data (np.ndarray): The building data
        """
        assert self.building_data.shape == data.shape
        self.building_data.from_numpy(data)

    @ti.kernel
    def compute_pass(
        self,
        baseline_template_ix: ti.i32,
        baseline_lab_ix: ti.i32,
        baseline_schedules_ix: ti.i32,
        retrofit_template_ix: ti.i32,
        retrofit_lab_ix: ti.i32,
        retrofit_schedules_ix: ti.i32,
        weather_scenario_ix: ti.i32,
    ):
        """
        Compute the demand for each end use for each hour for each year for each building for each simulation pass

        Args:
            baseline_template_ix (ti.i32): The index of the baseline building template
            baseline_lab_ix (ti.i32): The index of the baseline lab scenario
            baseline_schedules_ix (ti.i32): The index of the baseline building schedules
            retrofit_template_ix (ti.i32): The index of the retrofit building template
            retrofit_lab_ix (ti.i32): The index of the retrofit lab scenario
            retrofit_schedules_ix (ti.i32): The index of the retrofit building schedules
            weather_scenario_ix (ti.i32): The index of the weather scenario
        """
        # parallelize over simulation pass, year in pass, building in pass, end use in pass
        # could parallelize more granularly but not necessary (e.g. over time)
        for simulation_ix, year_ix, building_ix, enduse_ix in ti.ndrange(
            self.n_simulation_passes,
            self.n_years_per_sim,
            self.n_buildings,
            self.n_enduses,
        ):
            # get the upgrade year for the building
            building_scenario_upgrade_year = self.upgrade_years[
                simulation_ix, building_ix
            ]

            # select the baseline or upgrade scenario based on the year
            building_template_ix = baseline_template_ix
            building_lab_ix = baseline_lab_ix
            building_schedules_ix = baseline_schedules_ix
            if year_ix >= building_scenario_upgrade_year:
                building_template_ix = retrofit_template_ix
                building_lab_ix = retrofit_lab_ix
                building_schedules_ix = retrofit_schedules_ix

            # sum the demand for each hour into the corresponding end use
            current_year_ix = year_ix // self.n_years_per_eval
            next_year_ix = (
                current_year_ix + 1
                if current_year_ix < self.n_evals_per_building_scenario - 1
                else current_year_ix
            )
            phase = (year_ix % self.n_years_per_eval) / self.n_years_per_eval
            for hour_ix in range(self.n_timesteps):
                self.results[simulation_ix, year_ix, enduse_ix, hour_ix] += (
                    self.building_data[
                        weather_scenario_ix,
                        current_year_ix,
                        building_template_ix,
                        building_lab_ix,
                        building_schedules_ix,
                        building_ix,
                        enduse_ix,
                        hour_ix,
                    ]
                    * (1 - phase)
                    + self.building_data[
                        weather_scenario_ix,
                        next_year_ix,
                        building_template_ix,
                        building_lab_ix,
                        building_schedules_ix,
                        building_ix,
                        enduse_ix,
                        hour_ix,
                    ]
                    * phase
                )
        ti.sync()

    def run_pass(
        self,
        *,
        n_upgrades_executed_per_year: int,
        baseline_template_ix: int,
        baseline_lab_ix: int,
        baseline_schedules_ix: int,
        retrofit_template_ix: int,
        retrofit_lab_ix: int,
        retrofit_schedules_ix: int,
        weather_scenario_ix: int,
    ):
        """
        Run a single round of monte carlo upgrade sequencing

        Args:
            n_upgrades_executed_per_year (int): The number of buildings upgraded per year
            baseline_template_ix (int): The index of the baseline building template
            baseline_lab_ix (int): The index of the baseline lab scenario
            baseline_schedules_ix (int): The index of the baseline building schedules
            retrofit_template_ix (int): The index of the retrofit building template
            retrofit_lab_ix (int): The index of the retrofit lab scenario
            retrofit_schedules_ix (int): The index of the retrofit building schedules
            weather_scenario_ix (int): The index of the weather scenario
        """

        # Create random upgrade assignment sequences for each simulation pass
        upgrade_assignments = np.tile(
            np.arange(self.n_buildings), (self.n_simulation_passes, 1)
        )
        np.apply_along_axis(np.random.shuffle, 1, upgrade_assignments)
        upgrade_years_seed = upgrade_assignments // n_upgrades_executed_per_year

        # prep the taichi fields
        self.upgrade_years.from_numpy(upgrade_years_seed)
        self.results.fill(0.0)

        # run the pass
        self.compute_pass(
            baseline_template_ix=baseline_template_ix,
            baseline_lab_ix=baseline_lab_ix,
            baseline_schedules_ix=baseline_schedules_ix,
            retrofit_template_ix=retrofit_template_ix,
            retrofit_lab_ix=retrofit_lab_ix,
            retrofit_schedules_ix=retrofit_schedules_ix,
            weather_scenario_ix=weather_scenario_ix,
        )

        # get the results
        results = self.results.to_numpy()
        annual_results = results.sum(axis=(-1))

        # Create a multi-index for the DataFrame
        index = pd.MultiIndex.from_product(
            [range(self.n_simulation_passes), range(self.n_years_per_sim)],
            names=["Simulation Pass", "Year"],
        )

        # Convert the numpy array to a 2D array where each "row" will be a unique combination of simulation pass and year
        data_reshaped = annual_results.reshape(
            self.n_simulation_passes * self.n_years_per_sim, self.n_enduses
        )

        # Create the DataFrame
        df = pd.DataFrame(
            data_reshaped,
            index=index,
            columns=[
                f"End Use {i}" for i in range(self.n_enduses)
            ],  # TODO: naming of end uses
        )

        # Reset the index to turn the multi-index into regular columns
        df = df.reset_index()

        # Melt the DataFrame to get it in the desired long format
        melted_df = pd.melt(
            df,
            id_vars=["Simulation Pass", "Year"],
            var_name="End Use",
            value_name="Demand [J]",
        )
        melted_df["Building Retrofits Per Year"] = n_upgrades_executed_per_year
        melted_df["Retrofit Scenario"] = retrofit_template_ix
        melted_df["Lab Scenario"] = retrofit_lab_ix
        melted_df["Schedules Scenario"] = retrofit_schedules_ix
        melted_df["Weather Scenario"] = weather_scenario_ix

        return melted_df

    def run_retrofit_rates(
        self,
        *,
        baseline_template_ix: int,
        baseline_lab_ix: int,
        baseline_schedules_ix: int,
        retrofit_template_ix: int,
        retrofit_lab_ix: int,
        retrofit_schedules_ix: int,
        weather_scenario_ix: int,
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of retrofit rates for a single weather and retrofit scenario

        Args:
            baseline_template_ix (int): The index of the baseline building template
            baseline_lab_ix (int): The index of the baseline lab scenario
            baseline_schedules_ix (int): The index of the baseline building schedules
            retrofit_template_ix (int): The index of the retrofit building template
            retrofit_lab_ix (int): The index of the retrofit lab scenario
            retrofit_schedules_ix (int): The index of the retrofit building schedules
            weather_scenario_ix (int): The index of the weather scenario
            building_retrofit_rates (list[int]): The number of building upgrades executed per year
        """
        stacked_dfs = []
        for n_upgrades_executed_per_year in tqdm(
            building_retrofit_rates,
            desc="Retrofit Rates",
            position=4,
            leave=False,
        ):
            df = self.run_pass(
                n_upgrades_executed_per_year=n_upgrades_executed_per_year,
                baseline_template_ix=baseline_template_ix,
                baseline_schedules_ix=baseline_schedules_ix,
                baseline_lab_ix=baseline_lab_ix,
                retrofit_template_ix=retrofit_template_ix,
                retrofit_schedules_ix=retrofit_schedules_ix,
                retrofit_lab_ix=retrofit_lab_ix,
                weather_scenario_ix=weather_scenario_ix,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        return df

    def run_retrofit_scenarios(
        self,
        *,
        weather_scenario_ix: int = 0,
        baseline_template_ix: int = 0,
        baseline_lab_ix: int = 0,
        baseline_schedules_ix: int = 0,
        retrofit_lab_ix: int = 0,
        retrofit_schedules_ix: int = 0,
        retrofit_templates_ixs: list[tuple[str, int]] = [("Shallow", 1), ("Deep", 2)],
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of retrofit scenarios for a single weather scenario

        Args:
            weather_scenario_ix (int): The index of the weather scenario
            baseline_template_ix (int): The index of the baseline building template
            baseline_schedules_ix (int): The index of the baseline building schedules
            baseline_lab_ix (int): The index of the baseline lab scenario
            retrofit_schedules_ix (int): The index of the retrofit building schedules
            retrofit_lab_ix (int): The index of the retrofit lab scenario
            retrofit_templates_ix (list[tuple[str, int]]): The names and indices of the retrofit scenarios
            building_retrofit_rates (list[int]): The number of building upgrades executed per year
        """
        stacked_dfs = []
        for retrofit_name, retrofit_template_ix in tqdm(
            retrofit_templates_ixs,
            desc="Retrofit",
            position=3,
            leave=False,
        ):
            df = self.run_retrofit_rates(
                baseline_template_ix=baseline_template_ix,
                baseline_lab_ix=baseline_lab_ix,
                baseline_schedules_ix=baseline_schedules_ix,
                retrofit_template_ix=retrofit_template_ix,
                retrofit_lab_ix=retrofit_lab_ix,
                retrofit_schedules_ix=retrofit_schedules_ix,
                weather_scenario_ix=weather_scenario_ix,
                building_retrofit_rates=building_retrofit_rates,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        df["Retrofit Scenario"] = df["Retrofit Scenario"].map(
            {ix: name for name, ix in retrofit_templates_ixs}
        )
        return df

    def run_lab_scenarios(
        self,
        *,
        weather_scenario_ix: int = 0,
        baseline_template_ix: int = 0,
        baseline_lab_ix: int = 0,
        baseline_schedules_ix: int = 0,
        retrofit_schedules_ix: int = 0,
        retrofit_templates_ixs: list[tuple[str, int]] = [("Shallow", 1), ("Deep", 2)],
        retrofit_lab_ixs: list[tuple[str, int]] = [("Shallow", 1), ("Deep", 2)],
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of lab scenarios for a single weather scenario

        Args:
        """

        stacked_dfs = []
        for lab_name, retrofit_lab_ix in tqdm(
            retrofit_lab_ixs, desc="Lab", position=2, leave=False
        ):
            df = self.run_retrofit_scenarios(
                weather_scenario_ix=weather_scenario_ix,
                baseline_template_ix=baseline_template_ix,
                baseline_lab_ix=baseline_lab_ix,
                baseline_schedules_ix=baseline_schedules_ix,
                retrofit_templates_ixs=retrofit_templates_ixs,
                retrofit_lab_ix=retrofit_lab_ix,
                retrofit_schedules_ix=retrofit_schedules_ix,
                building_retrofit_rates=building_retrofit_rates,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        df["Lab Scenario"] = df["Lab Scenario"].map(
            {ix: name for name, ix in retrofit_lab_ixs}
        )
        return df

    def run_schedules_scenarios(
        self,
        *,
        weather_scenario_ix: int = 0,
        baseline_template_ix: int = 0,
        baseline_lab_ix: int = 0,
        baseline_schedules_ix: int = 0,
        retrofit_template_ixs: list[tuple[str, int]] = [
            ("Shallow", 1),
            ("Deep", 2),
        ],
        retrofit_lab_ixs: list[tuple[str, int]] = [("Shallow", 1), ("Deep", 2)],
        retrofit_schedules_ixs: list[tuple[str, int]] = [
            ("Setbacks", 1),
            ("Advanced", 2),
        ],
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of schedule scenarios for a single weather scenario

        Args:

        """
        stacked_dfs = []
        for schedules_name, schedules_upgrade_ix in tqdm(
            retrofit_schedules_ixs,
            desc="Schedules",
            position=1,
            leave=False,
        ):
            df = self.run_lab_scenarios(
                weather_scenario_ix=weather_scenario_ix,
                baseline_template_ix=baseline_template_ix,
                baseline_lab_ix=baseline_lab_ix,
                baseline_schedules_ix=baseline_schedules_ix,
                retrofit_templates_ixs=retrofit_template_ixs,
                retrofit_lab_ixs=retrofit_lab_ixs,
                retrofit_schedules_ix=schedules_upgrade_ix,
                building_retrofit_rates=building_retrofit_rates,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        df["Schedules Scenario"] = df["Schedules Scenario"].map(
            {ix: name for name, ix in retrofit_schedules_ixs}
        )
        return df

    def run_weather_scenarios(
        self,
        *,
        weather_scenario_ixs: list[tuple[str, int]] = [
            ("BAU", 0),
            ("Stable", 1),
            ("Runaway", 2),
        ],
        baseline_template_ix: int,
        baseline_lab_ix: int,
        baseline_schedules_ix: int,
        retrofit_template_ixs: list[tuple[str, int]] = [
            ("Shallow", 1),
            ("Deep", 2),
        ],
        retrofit_lab_ixs: list[tuple[str, int]] = [
            ("Shallow", 1),
            ("Deep", 2),
        ],
        retrofit_schedules_ixs: list[tuple[str, int]] = [
            ("Setbacks", 1),
            ("Advanced", 2),
        ],
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of weather scenarios

        Args:

        """
        stacked_dfs = []
        for weather_name, weather_scenario_ix in tqdm(
            weather_scenario_ixs,
            desc="Weather",
            position=0,
            leave=False,
        ):
            df = self.run_schedules_scenarios(
                weather_scenario_ix=weather_scenario_ix,
                baseline_template_ix=baseline_template_ix,
                baseline_lab_ix=baseline_lab_ix,
                baseline_schedules_ix=baseline_schedules_ix,
                retrofit_template_ixs=retrofit_template_ixs,
                retrofit_lab_ixs=retrofit_lab_ixs,
                retrofit_schedules_ixs=retrofit_schedules_ixs,
                building_retrofit_rates=building_retrofit_rates,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        df["Weather Scenario"] = df["Weather Scenario"].map(
            {ix: name for name, ix in weather_scenario_ixs}
        )

        described_df = (
            df.set_index(
                [
                    "Weather Scenario",
                    "Retrofit Scenario",
                    "Lab Scenario",
                    "Schedules Scenario",
                    "Building Retrofits Per Year",
                    "Simulation Pass",
                    "Year",
                    "End Use",
                ]
            )
            .groupby(
                [
                    "Weather Scenario",
                    "Retrofit Scenario",
                    "Lab Scenario",
                    "Schedules Scenario",
                    "Building Retrofits Per Year",
                    "Year",
                    "End Use",
                ]
            )
            .describe()
            .droplevel(0, axis=1)
            .reset_index()
        )
        return df, described_df


if __name__ == "__main__":
    import plotly.express as px
    import plotly.graph_objects as go

    make_npy()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    sim_manager = DemandScenarioProjector(
        n_simulation_passes=100,
        n_weather_scenarios=4,
        n_years_per_eval=5,
        n_evals_per_building_scenario=6,
        n_building_scenarios=3,
        n_lab_scenarios=3,
        n_schedules_scenarios=3,
        n_buildings=90,
        n_enduses=4,
        n_timesteps=8760,
    )
    sim_manager.load_from_np(np.load("data/all_scenarios.npy"))
    df, described_df = sim_manager.run_weather_scenarios(
        baseline_template_ix=0,
        baseline_lab_ix=0,
        baseline_schedules_ix=0,
        weather_scenario_ixs=[
            (climate_scenario.name, climate_scenario.value)
            for climate_scenario in ClimateScenarioTypeEnum
        ],
        retrofit_template_ixs=[
            (retrofit_scenario.name, retrofit_scenario.value)
            for retrofit_scenario in BuildingRetrofitLevelTypeEnum
        ],
        retrofit_lab_ixs=[
            (lab_scenario.name, lab_scenario.value)
            for lab_scenario in BuildingRetrofitLevelTypeEnum
        ],
        retrofit_schedules_ixs=[
            (schedules_scenario.name, schedules_scenario.value)
            for schedules_scenario in BuildingSchedulesTypeEnum
        ],
        building_retrofit_rates=[
            1,
            3,
            5,
            9,
        ],
    )
    df.to_hdf("data/retrofit_path_results.hdf", key="results", mode="w")
    described_df.to_hdf(
        "data/retrofit_path_results.hdf", key="described_results", mode="a"
    )
    exit()
    fig = px.line(
        described_df,
        x="Year",
        y="50%",
        error_y="std",
        facet_col="End Use",
        facet_col_wrap=2,
        color="Building Retrofits Per Year",
        line_dash="Retrofit Scenario",
        line_group="Schedules Scenario",
        labels={"50%": "Demand [J]"},
        symbol="Weather Scenario",
    )
    fig.show()
