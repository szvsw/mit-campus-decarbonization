import numpy as np
import pandas as pd
import taichi as ti
from tqdm.auto import tqdm


@ti.data_oriented
class DemandScenarioProjector:
    def __init__(
        self,
        *,
        n_simulation_passes: int,
        n_years_per_sim: int,
        n_weather_scenarios: int,
        n_evals_per_building_scenario: int,
        n_building_scenarios: int,
        n_buildings: int,
        n_enduses: int,
        n_timesteps: int,
    ):
        self.n_simulation_passes = n_simulation_passes
        self.n_years_per_sim = n_years_per_sim
        self.n_weather_scenarios = n_weather_scenarios
        self.n_evals_per_building_scenario = n_evals_per_building_scenario
        self.n_years_per_eval = n_years_per_sim // n_evals_per_building_scenario
        self.n_building_scenarios = n_building_scenarios
        self.n_buildings = n_buildings
        self.n_enduses = n_enduses
        self.n_timesteps = n_timesteps

        self.building_data = ti.field(
            dtype=ti.f32,
            shape=(
                n_weather_scenarios,
                n_evals_per_building_scenario,
                n_building_scenarios,
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

    def create_synthetic_building_results(self):
        """
        Create synthetic building results to use as a seed for the simulation

        This is a placeholder for the actual building results that would be used in the simulation.

        The synthetic building results are created by reading in a sample of building results from a file and then
        creating a set of building results that are similar to the sample but with some random variation.

        Args:
            None
        
        Returns:
            None
        """
        df = pd.read_hdf("data/5out.hdf", key="results")
        data_seed = df.values.T
        baseline_buildings = np.expand_dims(data_seed, axis=0).repeat(
            self.n_buildings, axis=0
        )
        baseline_buildings = baseline_buildings * (
            np.random.rand(self.n_buildings, 1, 1) * 10 + 0.95
        )
        upgraded_buildings = np.expand_dims(baseline_buildings, axis=0).repeat(
            self.n_building_scenarios, axis=0
        ) * np.linspace(1, 0.3, self.n_building_scenarios).reshape(-1, 1, 1, 1)
        final_buildings = np.expand_dims(upgraded_buildings, axis=0).repeat(
            self.n_evals_per_building_scenario, axis=0
        )
        final_buildings = np.expand_dims(final_buildings, axis=0).repeat(
            self.n_weather_scenarios, axis=0
        ) * np.linspace(0.5, 1.5, self.n_weather_scenarios).reshape(-1, 1, 1, 1, 1, 1)
        # TODO: add in weather multipliers and year multipliers
        self.building_data.from_numpy(final_buildings)

    @ti.kernel
    def compute_pass(
        self,
        baseline_scenario_ix: ti.i32,
        retrofit_upgrade_ix: ti.i32,
        weather_scenario_ix: ti.i32,
    ):
        """
        Compute the demand for each end use for each hour for each year for each building for each simulation pass

        Args:
            baseline_scenario_ix (int): The index of the baseline scenario
            retrofit_upgrade_ix (int): The index of the retrofit upgrade scenario
            weather_scenario_ix (int): The index of the weather scenario
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
            building_scenario_ix = baseline_scenario_ix
            if year_ix >= building_scenario_upgrade_year:
                building_scenario_ix = retrofit_upgrade_ix

            # sum the demand for each hour into the corresponding end use
            for hour_ix in range(self.n_timesteps):
                self.results[
                    simulation_ix, year_ix, enduse_ix, hour_ix
                ] += self.building_data[
                    weather_scenario_ix,
                    year_ix // self.n_years_per_eval,
                    building_scenario_ix,
                    building_ix,
                    enduse_ix,
                    hour_ix,
                ]
        ti.sync()

    def run_pass(
        self,
        *,
        n_upgrades_executed_per_year: int,
        baseline_scenario_ix: int,
        retrofit_upgrade_ix: int,
        weather_scenario_ix: int,
    ):
        """
        Run a single round of monte carlo upgrade sequencing

        Args:
            n_upgrades_executed_per_year (int): The number of building upgrades executed per year
            baseline_scenario_ix (int): The index of the baseline scenario
            retrofit_upgrade_ix (int): The index of the retrofit upgrade scenario
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
            baseline_scenario_ix=baseline_scenario_ix,
            retrofit_upgrade_ix=retrofit_upgrade_ix,
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
        melted_df["Retrofit Scenario"] = retrofit_upgrade_ix
        melted_df["Weather Scenario"] = weather_scenario_ix

        return melted_df

    def run_retrofit_rates(
        self,
        *,
        baseline_scenario_ix: int,
        retrofit_upgrade_ix: int,
        weather_scenario_ix: int,
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of retrofit rates for a single weather and retrofit scenario

        Args:
            baseline_scenario_ix (int): The index of the baseline scenario
            retrofit_upgrade_ix (int): The index of the retrofit upgrade scenario
            weather_scenario_ix (int): The index of the weather scenario
            building_retrofit_rates (list[int]): The number of building upgrades executed per year
        """
        stacked_dfs = []
        for n_upgrades_executed_per_year in tqdm(building_retrofit_rates):
            df = self.run_pass(
                n_upgrades_executed_per_year=n_upgrades_executed_per_year,
                baseline_scenario_ix=baseline_scenario_ix,
                retrofit_upgrade_ix=retrofit_upgrade_ix,
                weather_scenario_ix=weather_scenario_ix,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        return df

    def run_retrofit_scenarios(
        self,
        *,
        weather_scenario_ix: int = 0,
        baseline_scenario_ix: int = 0,
        retrofit_scenario_ixs: list[tuple[str, int]] = [("Shallow", 1), ("Deep", 2)],
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of retrofit scenarios for a single weather scenario

        Args:
            weather_scenario_ix (int): The index of the weather scenario
            baseline_scenario_ix (int): The index of the baseline scenario
            retrofit_scenario_ixs (list[tuple[str, int]]): The names and indices of the retrofit scenarios
            building_retrofit_rates (list[int]): The number of building upgrades executed per year
        """
        stacked_dfs = []
        for retrofit_name, retrofit_upgrade_ix in tqdm(retrofit_scenario_ixs):
            df = self.run_retrofit_rates(
                baseline_scenario_ix=baseline_scenario_ix,
                retrofit_upgrade_ix=retrofit_upgrade_ix,
                weather_scenario_ix=weather_scenario_ix,
                building_retrofit_rates=building_retrofit_rates,
            )
            stacked_dfs.append(df)
        df = pd.concat(stacked_dfs, axis=0)
        df["Retrofit Scenario"] = df["Retrofit Scenario"].map(
            {ix: name for name, ix in retrofit_scenario_ixs}
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
        baseline_scenario_ix: int,
        retrofit_scenario_ixs: list[tuple[str, int]] = [("Shallow", 1), ("Deep", 2)],
        building_retrofit_rates: list[int] = list(range(1, 7, 1)),
    ):
        """
        Run a set of weather scenarios

        Args:
            weather_scenario_ixs (list[tuple[str, int]]): The names and indices of the weather scenarios
            baseline_scenario_ix (int): The index of the baseline scenario
            retrofit_scenario_ixs (list[tuple[str, int]]): The names and indices of the retrofit scenarios
            building_retrofit_rates (list[int]): The number of building upgrades executed per year
        
        """
        stacked_dfs = []
        for weather_name, weather_scenario_ix in tqdm(weather_scenario_ixs):
            df = self.run_retrofit_scenarios(
                weather_scenario_ix=weather_scenario_ix,
                baseline_scenario_ix=baseline_scenario_ix,
                retrofit_scenario_ixs=retrofit_scenario_ixs,
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

    ti.init(arch=ti.gpu)

    sim_manager = DemandScenarioProjector(
        n_simulation_passes=10,
        n_years_per_sim=25,
        n_weather_scenarios=3,
        n_evals_per_building_scenario=5,
        n_building_scenarios=3,
        n_buildings=100,
        n_enduses=4,
        n_timesteps=8760,
    )
    sim_manager.create_synthetic_building_results()
    df, described_df = sim_manager.run_weather_scenarios(
        baseline_scenario_ix=0,
        weather_scenario_ixs=[
            ("BAU", 1),
            ("Stable", 0),
            ("Runaway", 2),
        ],
        building_retrofit_rates=[
            1,
            3,
            5,
            9,
        ],
        retrofit_scenario_ixs=[
            ("Shallow", 1),
            ("Deep", 2),
        ],
    )
    fig = px.line(
        described_df,
        x="Year",
        y="50%",
        error_y="std",
        facet_col="End Use",
        facet_col_wrap=2,
        color="Building Retrofits Per Year",
        line_dash="Retrofit Scenario",
        labels={"50%": "Demand [J]"},
        symbol="Weather Scenario",
    )
    fig.show()
