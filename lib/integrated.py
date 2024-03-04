import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import taichi as ti

from lib.utils import EndUseEnum


@ti.dataclass
class Demand:
    heating: ti.f32
    cooling: ti.f32
    lighting: ti.f32
    equipment: ti.f32
    vehicle: ti.f32

    @ti.func
    def elec(self):
        return self.lighting + self.equipment + self.vehicle


@ti.dataclass
class DistrictSystemCoP:
    heating: ti.f32
    cooling: ti.f32


@ti.dataclass
class DistrictSystemDemand:
    heating: ti.f32
    cooling: ti.f32

    @ti.func
    def load_thermal_demand(
        self, heating: ti.f32, cooling: ti.f32, cup_vs_5gdhc_frac: ti.f32
    ):
        frac = 1.0 - cup_vs_5gdhc_frac
        self.heating = heating * frac
        self.cooling = cooling * frac


@ti.dataclass
class DistrictSystem:
    cop: DistrictSystemCoP
    demand: DistrictSystemDemand
    net: ti.f32
    borefield: ti.f32
    elec: ti.f32

    @ti.func
    def load_thermal_demand(
        self, heating: ti.f32, cooling: ti.f32, cup_vs_5gdhc_frac: ti.f32
    ):
        self.demand.load_thermal_demand(heating, cooling, cup_vs_5gdhc_frac)

    @ti.func
    def calc_cop(self):
        # TODO: shared function for cop computation, possibly also depends on net, reservoir?
        pass

    @ti.func
    def compute(self):
        self.net = self.demand.heating - self.demand.cooling
        self.calc_cop()
        heating_elec = self.demand.heating * self.cop.heating
        cooling_elec = self.demand.cooling * self.cop.cooling
        self.elec = heating_elec + cooling_elec


@ti.dataclass
class ThermalRouter:
    cup_vs_5gdhc_frac: ti.f32


@ti.dataclass
class CUPCarbonFactors:
    heating: ti.f32
    cooling: ti.f32
    electricity: ti.f32


@ti.dataclass
class CUPDemand:
    heating: ti.f32
    cooling: ti.f32
    electricity: ti.f32

    @ti.func
    def load_thermal_demand(
        self, heating: ti.f32, cooling: ti.f32, cup_vs_5gdhc_frac: ti.f32
    ):
        self.heating = heating * cup_vs_5gdhc_frac
        self.cooling = cooling * cup_vs_5gdhc_frac


@ti.dataclass
class CUPCarbon:
    stored: ti.f32
    emitted: ti.f32


@ti.dataclass
class CUP:
    demand: CUPDemand
    carbon_factors: CUPCarbonFactors
    carbon_capture_capacity: ti.f32
    turbine_capacity: ti.f32
    carbon: CUPCarbon

    @ti.func
    def load_thermal_demand(
        self, heating: ti.f32, cooling: ti.f32, cup_vs_5gdhc_frac: ti.f32
    ):
        self.demand.load_thermal_demand(heating, cooling, cup_vs_5gdhc_frac)

    @ti.func
    def compute(self):
        # TODO: should this depend on capacity somehow?
        cup_elec_carbon = self.demand.electricity * self.carbon_factors.electricity
        cup_heating_carbon = self.demand.heating * self.carbon_factors.heating
        cup_cooling_carbon = self.demand.cooling * self.carbon_factors.cooling
        cup_carbon = cup_elec_carbon + cup_heating_carbon + cup_cooling_carbon
        self.carbon.stored = ti.min(cup_carbon, self.carbon_capture_capacity)
        self.carbon.emitted = cup_carbon - self.carbon.stored


@ti.dataclass
class CleanElectricity:
    uReactor_capacity: ti.f32
    PV_capacity: ti.f32
    deep_geo_capacity: ti.f32

    @ti.func
    def capacity(self):
        return self.uReactor_capacity + self.PV_capacity + self.deep_geo_capacity


@ti.dataclass
class GridImpEx:
    imp: ti.f32
    ex: ti.f32


@ti.dataclass
class GridCostFactors:
    imp: ti.f32
    ex: ti.f32


@ti.dataclass
class Grid:
    carbon_factor: ti.f32
    costs: GridCostFactors
    impex: GridImpEx
    emitted: ti.f32
    cost: ti.f32

    @ti.func
    def compute(self):
        self.emitted = self.impex.imp * self.carbon_factor
        self.cost = self.impex.imp * self.costs.imp - self.impex.ex * self.costs.ex


@ti.dataclass
class Battery:
    capacity: ti.f32
    state_of_charge: ti.f32
    charge_rate: ti.f32
    discharge_rate: ti.f32


@ti.dataclass
class Timestep:
    demand: Demand
    district_system: DistrictSystem
    thermal_router: ThermalRouter
    cup: CUP
    total_electricity: ti.f32
    clean_electricity: CleanElectricity
    grid: Grid
    battery: Battery
    emissions: ti.f32

    @ti.func
    def thermal_routing(self):
        # heating and cooling
        self.cup.load_thermal_demand(
            self.demand.heating,
            self.demand.cooling,
            self.thermal_router.cup_vs_5gdhc_frac,
        )
        self.district_system.load_thermal_demand(
            self.demand.heating,
            self.demand.cooling,
            self.thermal_router.cup_vs_5gdhc_frac,
        )
        self.district_system.compute()

    @ti.func
    def electricity_routing(self):
        self.total_electricity = self.demand.elec() + self.district_system.elec

        # clean electricity capacity
        clean_elec_cap = self.clean_electricity.capacity()
        elec_net = self.total_electricity - clean_elec_cap
        elec_surplus = ti.max(0, -elec_net)
        elec_deficit = ti.max(0, elec_net)
        # TODO: battery charge/discharge, grid arbitrage, sales, etc
        self.cup.demand.electricity = ti.min(self.cup.turbine_capacity, elec_deficit)
        self.grid.impex.imp = ti.max(0, elec_deficit - self.cup.demand.electricity)
        self.grid.impex.ex = elec_surplus

    @ti.func
    def compute(self):
        # heating and cooling
        self.thermal_routing()

        # electricity demand
        self.electricity_routing()

        # cup carbon
        self.cup.compute()

        # grid carbon
        self.grid.compute()

        # total carbon
        self.emissions = self.cup.carbon.emitted + self.grid.emitted


# def timestep_1hr(
#     total_heating_kWh: float,
#     total_cooling_kWh: float,
#     total_equipment_kWh: float,
#     total_lighting_kWh: float,
#     total_vehicle_kWh: float,
#     cup_vs_5gdhc_frac: float,
#     cup_heating_carbon_factor_kgCO2e_per_kWh: float,
#     cup_cooling_carbon_factor_kgCO2e_per_kWh: float,
#     cup_electricity_carbon_factor_kgCO2e_per_kWh: float,
#     cup_carbon_storage_capacity_kgCO2e: float,
#     cup_turbine_capacity_kW: float,
#     district_heating_cop: float,
#     district_cooling_cop: float,
#     uReactor_capacity_kWh: float,
#     PV_capacity_kWh: float,
#     deep_geo_capacity_kWh: float,
#     battery_capacity_kWh: float,
#     battery_state_of_charge_kWh: float,
#     battery_charge_rate_kW: float,
#     battery_discharge_rate_kW: float,
#     grid_carbon_factor_kgCO2e_per_kWh: float,
# ):
#     # heating and cooling
#     cup_heating_kWh = total_heating_kWh * cup_vs_5gdhc_frac
#     cup_cooling_kWh = total_cooling_kWh * cup_vs_5gdhc_frac
#     district_heating_kWh = total_heating_kWh - cup_heating_kWh
#     district_cooling_kWh = total_cooling_kWh - cup_cooling_kWh

#     # district system performance
#     # TODO: shared function for cop computation, possibly also depends on net, reservoir?
#     net_demand = district_heating_kWh - district_cooling_kWh
#     district_heating_elec_kWh = district_heating_kWh * district_heating_cop
#     district_cooling_elec_kWh = district_cooling_kWh * district_cooling_cop

#     # electricity demand
#     total_elec_kWh = (
#         total_equipment_kWh
#         + total_lighting_kWh
#         + total_vehicle_kWh
#         + district_cooling_elec_kWh
#         + district_heating_elec_kWh
#     )

#     # clean electricity capacity
#     clean_elec_cap_kWh = uReactor_capacity_kWh + PV_capacity_kWh + deep_geo_capacity_kWh
#     elec_net_kWh = total_elec_kWh - clean_elec_cap_kWh
#     elec_surplus_kWh = max(0, -elec_net_kWh)  # TODO: surplus sales/storage
#     elec_deficit_kWh = max(0, elec_net_kWh)
#     # TODO: battery charging/discharging
#     # TODO: prioritiziation between cup and grid based off carbon, cost, etc
#     cup_elec_kWh = min(cup_turbine_capacity_kW, elec_deficit_kWh)
#     grid_elec_import_kWh = max(0, elec_deficit_kWh - cup_elec_kWh)

#     # cup carbon
#     # TODO: shared function for carbon factor computation, possibly also depends on cap?
#     cup_electricity_carbon_kgCO2e = (
#         cup_elec_kWh * cup_electricity_carbon_factor_kgCO2e_per_kWh
#     )
#     cup_heating_carbon_kgCO2e = (
#         cup_heating_kWh * cup_heating_carbon_factor_kgCO2e_per_kWh
#     )
#     cup_cooling_carbon_kgCO2e = (
#         cup_cooling_kWh * cup_cooling_carbon_factor_kgCO2e_per_kWh
#     )
#     cup_carbon_kgCO2e = (
#         cup_electricity_carbon_kgCO2e
#         + cup_heating_carbon_kgCO2e
#         + cup_cooling_carbon_kgCO2e
#     )
#     cup_carbon_storage_kgCO2e = min(
#         cup_carbon_kgCO2e, cup_carbon_storage_capacity_kgCO2e
#     )
#     cup_emissions_kgCO2e = cup_carbon_kgCO2e - cup_carbon_storage_kgCO2e

#     # grid carbon
#     grid_emissions_kgCO2e = grid_elec_import_kWh * grid_carbon_factor_kgCO2e_per_kWh

#     # total carbon
#     emissions_kgCO2e = cup_emissions_kgCO2e + grid_emissions_kgCO2e


def sankey_ex():
    df = pd.DataFrame(
        [
            ("Building 1", "Heating", 100, "kWh"),
            ("Building 1", "Cooling", 100, "kWh"),
            ("Building 1", "Lighting + Equipment", 50, "kWh"),
            ("Building 2", "Heating", 100, "kWh"),
            ("Building 2", "Cooling", 100, "kWh"),
            ("Building 2", "Lighting + Equipment", 50, "kWh"),
            ("Building 3", "Heating", 100, "kWh"),
            ("Building 3", "Cooling", 100, "kWh"),
            ("Building 3", "Lighting + Equipment", 50, "kWh"),
            ("Building 4", "Heating", 100, "kWh"),
            ("Building 4", "Cooling", 100, "kWh"),
            ("Building 4", "Lighting + Equipment", 50, "kWh"),
            ("Building 5", "Heating", 100, "kWh"),
            ("Building 5", "Cooling", 100, "kWh"),
            ("Building 5", "Lighting + Equipment", 50, "kWh"),
            ("Cooling", "5GDC", 100, "kWh"),
            ("Cooling", "CUP Cooling (ES Gas)", 400, "kWh"),
            ("Heating", "5GDH", 100, "kWh"),
            ("Heating", "CUP Heating (ES Gas)", 400, "kWh"),
            ("Vehicles", "Electricity", 100, "kWh"),
            ("Lighting + Equipment", "Electricity", 250, "kWh"),
            ("5GDC", "Electricity", 100, "kWh"),
            ("5GDH", "Electricity", 100, "kWh"),
            ("CUP Cooling (ES Gas)", "Carbon Emissions", 100, "kgCO2e"),
            ("CUP Heating (ES Gas)", "Carbon Emissions", 100, "kgCO2e"),
            ("CUP Cooling (ES Gas)", "Local Carbon Capture", 100, "kgCO2e"),
            ("CUP Heating (ES Gas)", "Local Carbon Capture", 100, "kgCO2e"),
            ("Electricity", "uReactor 1", 100, "kWh"),
            ("Electricity", "uReactor 2", 100, "kWh"),
            ("Electricity", "Deep Geothermal", 100, "kWh"),
            ("Electricity", "PV", 100, "kWh"),
            ("Electricity", "CUP Turbine (ES Gas)", 100, "kWh"),
            ("Electricity", "Local Storage (draining)", 100, "kWh"),
            ("Electricity", "Imported (ES Electric)", 100, "kWh"),
            ("Imported (ES Electric)", "Carbon Emissions", 100, "kgCO2e"),
            ("CUP Turbine (ES Gas)", "Carbon Emissions", 100, "kgCO2e"),
            ("CUP Turbine (ES Gas)", "Local Carbon Capture", 100, "kgCO2e"),
        ],
        columns=["Source", "Target", "Value", "Unit"],
    )

    node_names = list(set(df["Source"].unique()) | set(df["Target"].unique()))
    sources = df["Source"].map(lambda x: node_names.index(x))
    targets = df["Target"].map(lambda x: node_names.index(x))
    values = df["Value"]
    units = df["Unit"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=node_names,
                    pad=25,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=units,
                ),
            )
        ]
    )
    fig.show()


@ti.data_oriented
class ScenarioComputer:
    timesteps: ti.StructField

    def __init__(
        self,
        heating: np.ndarray,
        cooling: np.ndarray,
        lighting: np.ndarray,
        equipment: np.ndarray,
        vehicle: np.ndarray,
    ):
        # weather scenario, retrofit scenario, lab scenario, schedule scenario, retrofit rate, years, timestep
        demand = {
            "heating": heating,
            "cooling": cooling,
            "lighting": lighting,
            "equipment": equipment,
            "vehicle": vehicle,
        }
        assert heating.shape == (4, 3, 3, 3, 3, 26, 8760)
        self.timesteps = Timestep.field(shape=heating.shape)
        self.timesteps.demand.from_numpy(demand)

    @ti.kernel
    def compute(self):
        for i in ti.grouped(self.timesteps):
            self.timesteps[i].compute()


def dummy_district_scenario():
    district_cop = []
    # configure thermal router
    thermal_router = []
    for i in range(3):

        district_cop.append(
            {
                "heating": (
                    np.ones((scenarios.timesteps.shape)).astype(np.float32)
                    * (3.0 if i <= 1 else 3.5)
                ),
                "cooling": (
                    np.ones((scenarios.timesteps.shape)).astype(np.float32)
                    * (4.0 if i <= 1 else 4.5)
                ),
            }
        )

        frac = np.ones((scenarios.timesteps.shape)).astype(np.float32)
        if i == 0:
            pass
        else:
            n_years = 20 - i * 5
            total_years = heating.shape[-2]
            assert total_years == 26
            frac[:, :, :, :, :, :10] = 1.0
            frac[:, :, :, :, :, :, 10 : 10 + n_years] = np.linspace(1.0, 0.0, n_years)
            frac[:, :, :, :, :, :, 10 + n_years :] = 0.0

        thermal_router.append({"cup_vs_5gdhc_frac": frac})
    return district_cop, thermal_router


def dummy_cup_carbon_factors():
    cup_carbon_factors = {
        "heating": np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0.181,
        "cooling": np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0.181,
        "electricity": np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0.181,
    }
    return cup_carbon_factors


def dummy_cup_turbine_capacity():
    cup_turbine_capacity = (
        np.ones((scenarios.timesteps.shape)).astype(np.float32) * 40000
    )
    return cup_turbine_capacity


def dummy_cup_carbon_capture_capacity():
    cup_carbon_capture_capacities = []
    for i in range(3):
        cup_carbon_capture_capacity = np.ones((scenarios.timesteps.shape)).astype(
            np.float32
        ) * (3000 if i <= 1 else 4000)
        if i == 0:
            cup_carbon_capture_capacity *= 0
        if i == 1:
            cup_carbon_capture_capacity[:, :, :, :, :, :, :15] = 0
        if i == 2:
            cup_carbon_capture_capacity[:, :, :, :, :, :, :10] = 0
        cup_carbon_capture_capacities.append(cup_carbon_capture_capacity)
    return cup_carbon_capture_capacities


def dummy_battery():

    capacities = []
    charge_rates = []
    discharge_rates = []
    for i in range(3):
        battery_capacity = np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0
        battery_charge_rate = (
            np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0
        )
        battery_discharge_rate = (
            np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0
        )
        capacities.append(battery_capacity)
        charge_rates.append(battery_charge_rate)
        discharge_rates.append(battery_discharge_rate)
    return capacities, charge_rates, discharge_rates


def dummy_grid_scenarios():
    costs = []
    emissions = []
    for i in range(3):
        # TODO: better costs
        grid_cost_factors = {
            "imp": np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0.1,
            "ex": np.ones((scenarios.timesteps.shape)).astype(np.float32) * 0.05,
        }
        grid_carbon_factor = np.ones((scenarios.timesteps.shape)).astype(np.float32) * (
            0.216
        )
        if i == 0:
            year_decay = np.linspace(1.0, 0.0, scenarios.timesteps.shape[-2]).reshape(
                -1, 1
            )
            base_shape = len(scenarios.timesteps.shape[:-2])
            for i in range(base_shape):
                year_decay = year_decay[np.newaxis, ...]
            grid_carbon_factor *= year_decay
        if i > 0:
            finishing_year = 5 + 5 * i
            year_decay = np.linspace(
                1.0, 0.0, scenarios.timesteps.shape[-2] - finishing_year
            )

            year_decay = np.concatenate([year_decay, np.zeros((finishing_year,))])
            year_decay = year_decay.reshape(-1, 1)
            base_shape = len(scenarios.timesteps.shape[:-2])
            for i in range(base_shape):
                year_decay = year_decay[np.newaxis, ...]
                grid_carbon_factor *= year_decay
        costs.append(grid_cost_factors)
        emissions.append(grid_carbon_factor)
    return costs, emissions


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    from tqdm import tqdm

    end_uses_J = np.load("data/hourly_results.npy")
    end_uses_kWh = end_uses_J * 2.77778e-7
    heating = end_uses_kWh[:, :, :, :, :, :, :, EndUseEnum.heating]
    cooling = end_uses_kWh[:, :, :, :, :, :, :, EndUseEnum.cooling]
    lighting = end_uses_kWh[:, :, :, :, :, :, :, EndUseEnum.lighting]
    equipment = end_uses_kWh[:, :, :, :, :, :, :, EndUseEnum.equipment]
    scenarios = ScenarioComputer(
        heating=heating,
        cooling=cooling,
        lighting=lighting,
        equipment=equipment,
        vehicle=np.zeros_like(heating),
    )
    cup_carbon_factors = dummy_cup_carbon_factors()
    cup_turbine_capacity = dummy_cup_turbine_capacity()

    district_cop, thermal_router = dummy_district_scenario()

    cup_carbon_capture_capacity = dummy_cup_carbon_capture_capacity()

    clean_electricity = {
        "uReactor_capacity": np.random.rand(*(scenarios.timesteps.shape)).astype(
            np.float32
        )
        * 0,
        "PV_capacity": np.random.rand(*(scenarios.timesteps.shape)).astype(np.float32)
        * 0,
        "deep_geo_capacity": np.random.rand(*(scenarios.timesteps.shape)).astype(
            np.float32
        )
        * 0,
    }
    grid_cost_factors, grid_carbon_factor = dummy_grid_scenarios()
    battery_capacity, battery_charge_rate, battery_discharge_rate = dummy_battery()

    # shared data
    scenarios.timesteps.cup.carbon_factors.from_numpy(cup_carbon_factors)
    scenarios.timesteps.cup.turbine_capacity.from_numpy(cup_turbine_capacity)
    annual_emissions = np.zeros(
        shape=(3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 26), dtype=np.float32
    )
    for grid_scenario in tqdm(
        range(3),
        desc="Grid",
        position=0,
        leave=False,
    ):
        scenarios.timesteps.grid.costs.from_numpy(grid_cost_factors[grid_scenario])
        scenarios.timesteps.grid.carbon_factor.from_numpy(
            grid_carbon_factor[grid_scenario]
        )
        for ccs_scenario in tqdm(
            range(3),
            desc="CCS",
            position=1,
            leave=False,
        ):
            scenarios.timesteps.cup.carbon_capture_capacity.from_numpy(
                cup_carbon_capture_capacity[ccs_scenario]
            )
            for battery_scenario in tqdm(
                range(3),
                desc="Battery",
                position=2,
                leave=False,
            ):
                scenarios.timesteps.battery.capacity.from_numpy(
                    battery_capacity[battery_scenario]
                )
                scenarios.timesteps.battery.charge_rate.from_numpy(
                    battery_charge_rate[battery_scenario]
                )
                scenarios.timesteps.battery.discharge_rate.from_numpy(
                    battery_discharge_rate[battery_scenario]
                )
                for deep_geo_scenario in tqdm(
                    range(3),
                    desc="Deep Geo",
                    position=3,
                    leave=False,
                ):
                    for ureactor_scenario in tqdm(
                        range(3),
                        desc="uReactor",
                        position=4,
                        leave=False,
                    ):
                        scenarios.timesteps.clean_electricity.from_numpy(
                            clean_electricity
                        )
                        for district_scenario in tqdm(
                            range(3),
                            desc="District System",
                            position=5,
                            leave=False,
                        ):
                            scenarios.timesteps.district_system.cop.from_numpy(
                                district_cop[district_scenario]
                            )
                            scenarios.timesteps.thermal_router.from_numpy(
                                thermal_router[district_scenario]
                            )
                            scenarios.compute()
                            emissions = scenarios.timesteps.emissions.to_numpy()
                            emissions = emissions.sum(axis=-1)
                            assert (
                                annual_emissions[
                                    grid_scenario,
                                    ccs_scenario,
                                    battery_scenario,
                                    deep_geo_scenario,
                                    ureactor_scenario,
                                    district_scenario,
                                ].shape
                                == emissions.shape
                            )
                            annual_emissions[
                                grid_scenario,
                                ccs_scenario,
                                battery_scenario,
                                deep_geo_scenario,
                                ureactor_scenario,
                                district_scenario,
                            ] = emissions
                            ti.sync()
    import plotly.graph_objects as go

    # flatten all but the last axis of annual_emissions
    print(annual_emissions.shape)
    annual_emissions = annual_emissions.reshape(-1, annual_emissions.shape[-1])
    fig = go.Figure()
    for i in range(annual_emissions.shape[0]):
        if np.random.rand() < 0.05:
            fig.add_trace(
                go.Scatter(x=np.arange(26), y=annual_emissions[i], mode="lines")
            )

    fig.update_traces(
        line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
    )
    fig.show()
