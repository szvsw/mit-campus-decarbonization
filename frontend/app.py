import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

try:
    from frontend import frontend_settings
except:
    print("appending path...")
    logger.info("appending path...")
    sys.path.append("/mount/src/mit-campus-decarbonization")
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from frontend import frontend_settings as settings
from lib.client import client
from lib.models import (
    Building,
    BuildingSimulationResult,
    DemandScenario,
    DemandScenarioBuilding,
    PowerPlant,
    PowerPlantScenario,
    engine,
)

st.set_page_config(layout="wide", page_title="MIT Decarbonization")


@st.cache_data
def get_all_buildings():
    df = pd.DataFrame(client.table("Building").select("*").execute().data).set_index(
        "id"
    )
    # create csv bytes
    csv = df.to_csv().encode()
    return df, csv


@st.cache_data
def get_all_buildings_v2():
    with Session(engine) as session:
        stmt = select(Building)
        buildings = session.exec(stmt).all()
        df = pd.DataFrame([b.model_dump() for b in buildings]).set_index("id")
        csv = df.to_csv().encode()
    return df, csv


@st.cache_data
def get_building_scenarios(building_id: int):
    ids = (
        client.table("DemandScenarioBuilding")
        .select("id, demand_scenario_id")
        .eq("building_id", building_id)
        .execute()
        .data
    )
    scenario_ids = [d["demand_scenario_id"] for d in ids]
    results_ids = [d["id"] for d in ids]

    return scenario_ids, results_ids


@st.cache_data
def get_building_scenarios_v2(building_id: int):
    with Session(engine) as session:
        stmt = select(DemandScenarioBuilding).where(
            DemandScenarioBuilding.building_id == building_id
        )
        scenarios = session.exec(stmt).all()
        scenario_ids = [s.demand_scenario_id for s in scenarios]
        results_ids = [s.id for s in scenarios]

    return scenario_ids, results_ids


@st.cache_data
def get_scenarios():
    return client.table("DemandScenario").select("*").execute().data


def create_demand_scenario(name: str):
    with Session(engine) as session:
        scenario = DemandScenario(name=name)
        session.add(scenario)
        session.commit()
        st.cache_data.clear()
        st.rerun()


def create_power_plant(name: str):
    with Session(engine) as session:
        scenario = PowerPlant(name=name)
        session.add(scenario)
        session.commit()
        get_power_plants.clear()
        st.rerun()


def create_power_plant_scenario(name: str, power_plant_id: int, df: pd.DataFrame):
    with Session(engine) as session:
        scenario = PowerPlantScenario(
            name=name,
            power_plant_id=power_plant_id,
            emissions_factors=df["emissions_factors"].values,
            cost_factors=df["cost_factors"].values,
            capacities=df["capacities"].values,
        )
        session.add(scenario)
        session.commit()
        get_power_plant_scenarios.clear()
        st.rerun()


@st.cache_data
def get_scenario_results(scenario_id: int):
    ids = (
        client.table("DemandScenarioBuilding")
        .select("id")
        .eq("demand_scenario_id", scenario_id)
        .execute()
        .data
    )
    results_ids = [d["id"] for d in ids]
    results = (
        client.table("BuildingSimulationResult")
        .select("*")
        .in_("id", results_ids)
        .execute()
        .data
    )
    dfs = []
    for result in results:
        result["heating"] = json.loads(result["heating"])
        result["cooling"] = json.loads(result["cooling"])
        result["lighting"] = json.loads(result["lighting"])
        result["equipment"] = json.loads(result["equipment"])
        df = pd.DataFrame(
            {
                "heating": result["heating"],
                "cooling": result["cooling"],
                "lighting": result["lighting"],
                "equipment": result["equipment"],
            }
        )
        df["Timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")
        df = df.set_index("Timestamp")
        df["result_id"] = result["id"]
        df = df.set_index("result_id", append=True)
        dfs.append(df)
    if len(dfs) == 0:
        return None, None, None
    df_buildings = pd.concat(dfs)
    df = df_buildings.groupby("Timestamp").sum()
    df.columns = [x.capitalize() for x in df.columns]
    df = df.reset_index("Timestamp")
    df_melted = df.melt(
        id_vars=["Timestamp"], var_name="End Use", value_name="Energy [J]"
    )
    return df, df_melted, df_buildings


@st.cache_data
def get_scenario_building_result(scenario_id: int, building_id: int):
    ids = (
        client.table("DemandScenarioBuilding")
        .select("id")
        .eq("demand_scenario_id", scenario_id)
        .eq("building_id", building_id)
        .execute()
        .data
    )
    results_ids = [d["id"] for d in ids]
    results = (
        client.table("BuildingSimulationResult")
        .select("*")
        .in_("id", results_ids)
        .execute()
        .data
    )
    df = pd.DataFrame(
        {
            "heating": json.loads(results[0]["heating"]),
            "cooling": json.loads(results[0]["cooling"]),
            "lighting": json.loads(results[0]["lighting"]),
            "equipment": json.loads(results[0]["equipment"]),
        }
    )
    df["Timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")
    df.columns = [x.capitalize() for x in df.columns]
    df_melted = df.melt(
        id_vars=["Timestamp"], var_name="End Use", value_name="Energy [J]"
    )

    return df, df_melted


@st.cache_data
def get_power_plants() -> list[PowerPlant]:
    with Session(engine) as session:
        stmt = select(PowerPlant).options(
            selectinload(PowerPlant.power_plant_scenarios)
        )
        power_plants = session.exec(stmt).all()

        return [(p.model_dump()) for p in power_plants]


@st.cache_data
def get_power_plant_scenarios(power_plant_id: int) -> list[PowerPlantScenario]:
    with Session(engine) as session:
        stmt = select(PowerPlantScenario).where(
            PowerPlantScenario.power_plant_id == power_plant_id
        )
        power_plant_scenarios = session.exec(stmt).all()

        return [(p.model_dump(), p.to_df()) for p in power_plant_scenarios]


ENDUSE_PASTEL_COLORS = {
    "Heating": "#FF7671",
    "Cooling": "#6D68E6",
    "Lighting": "#FFD700",
    "Equipment": "#90EE90",
}


def render_title():
    st.title("MIT Decarbonization")


def render_buildings():
    all_buildings_df, all_buildings_csv = get_all_buildings_v2()
    st.download_button(
        "Download all building metadata",
        all_buildings_csv,
        "buildings_metadata.csv",
        "Download all buildings",
        use_container_width=True,
        type="primary",
    )
    building_id = st.selectbox(
        "Building",
        all_buildings_df.index,
        format_func=lambda x: all_buildings_df.loc[x, "name"],
        help="Select a building to view its data",
    )
    building = all_buildings_df.loc[building_id]
    scenario_ids, results_ids = get_building_scenarios(building_id)
    all_scenarios = get_scenarios()
    filtered_scenarios = [s for s in all_scenarios if s["id"] in scenario_ids]
    l, r = st.columns(2)
    with l:
        scenario_id = st.selectbox(
            "Building Demand Scenario",
            scenario_ids,
            format_func=lambda x: [
                s["name"] for s in filtered_scenarios if s["id"] == x
            ][0],
            help="Select a demand scenario to view its data",
        )
        result, result_melted = get_scenario_building_result(scenario_id, building_id)
        fig = px.line(
            result_melted,
            x="Timestamp",
            y="Energy [J]",
            color="End Use",
            title=f"Building {building['name']} Energy Use",
            color_discrete_map=ENDUSE_PASTEL_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True)

    with r:
        st.dataframe(building)


@st.cache_data
def encode_csv(df: pd.DataFrame) -> str:
    return df.to_csv().encode()


def render_building_scenarios():
    all_scenarios = get_scenarios()
    scenario = st.selectbox(
        "Demand Scenario",
        all_scenarios,
        format_func=lambda x: x["name"],
        help="Select a demand scenario to view its data",
    )
    df, df_melted, df_buildings = get_scenario_results(scenario["id"])
    if df is None:
        st.warning("No results found for this scenario!")
        render_create_demand_scenario()
        return

    l, r = st.columns(2)
    with l:
        st.download_button(
            "Download scenario results",
            encode_csv(df),
            f"{scenario['name']}_results.csv",
            "Download scenario results",
            use_container_width=True,
            type="primary",
        )
    with r:
        st.download_button(
            "Download scenario buildings results",
            encode_csv(df_buildings),
            f"{scenario['name']}_buildings_results.csv",
            "Download scenario buildings results",
            use_container_width=True,
            type="primary",
        )
    fig = px.line(
        df_melted,
        x="Timestamp",
        y="Energy [J]",
        color="End Use",
        title=f"{scenario['name']} Demand Scenario",
        color_discrete_map=ENDUSE_PASTEL_COLORS,
    )
    st.plotly_chart(fig, use_container_width=True)
    render_create_demand_scenario()


def render_create_demand_scenario():
    st.divider()
    st.markdown("### Create a new demand scenario")
    name = st.text_input(
        "Name", key="create_demand_scenario_name", help="Name of the scenario"
    )
    if st.button("Create", disabled=name == "", key="create_demand_scenario_button"):
        create_demand_scenario(name)


def render_create_power_plant():
    st.divider()
    st.markdown("### New Power Plant")
    st.markdown("Create a new power plant type.")
    name = st.text_input(
        "Name", key="create_power_plant_name", help="Name of the power plant type"
    )
    if st.button(
        "Create",
        disabled=name == "",
        key="create_power_plant_button",
        type="primary",
        use_container_width=True,
    ):
        create_power_plant(name)


# TODO: use pandera, import UploadedFile type annotation
@st.cache_data
def validate_power_plant_scenario_csv(csv_file) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_file)
        # assert "Timestamp" in df.columns
        assert "capacities" in df.columns
        assert "emissions_factors" in df.columns
        assert "cost_factors" in df.columns
        assert len(df) == 8760
        assert df["capacities"].dtype in [int, float]
        assert df["emissions_factors"].dtype in [int, float]
        assert df["cost_factors"].dtype in [int, float]
        # assert df["Timestamp"].dtype == "datetime64[ns]"
        return df
    except Exception as e:
        return None


def render_create_power_plant_scenario(pp: PowerPlant):
    st.divider()
    st.markdown("### New Power Plant Scenario")
    st.markdown(f"Create a new scenario for `{pp.name}` power plant type")
    name = st.text_input(
        "Name", key="create_power_plant_scenario_name", help="Name of the scenario"
    )
    csv_file = st.file_uploader(
        "Upload a CSV file with the scenario data",
        type=["csv"],
        accept_multiple_files=False,
        key="upload_power_plant_scenario_data",
    )
    df = None
    if csv_file:
        df = validate_power_plant_scenario_csv(csv_file)
        if df is None:
            st.error(
                "Invalid CSV file. Please check the file format and try again.  The CSV must have three columns: 'capacities', 'emissions_factors', and 'cost_factors' and 8760 rows."
            )

    disabled = name == "" or df is None
    if st.button(
        "Create",
        disabled=disabled,
        key="create_power_plant_scenario_button",
        use_container_width=True,
        type="primary",
    ):
        create_power_plant_scenario(name, pp.id, df)


def render_power_plants():
    l, r = st.columns(2)
    chart_container = st.container()
    create_plant_container, create_plant_scenario_container = st.columns(2)

    with create_plant_container:
        render_create_power_plant()

    # get the power plants
    power_plants = [PowerPlant.model_validate(p) for p in get_power_plants()]

    # early return if no power plants
    if len(power_plants) == 0:
        st.warning("No power plants found!")
        return

    # select a plant
    with l:
        pp = st.selectbox(
            "Power Plant",
            power_plants,
            format_func=lambda x: x.name,
            help="Select a power plant to view its data",
        )

    with create_plant_scenario_container:
        render_create_power_plant_scenario(pp)

    # get scenarios
    pp_scenarios = [
        (PowerPlantScenario.model_validate(p))
        for p, df in get_power_plant_scenarios(pp.id)
    ]

    # early return if no scenarios
    if len(pp_scenarios) == 0:
        with r:
            st.warning("No scenarios found for this power plant!")
        return

    # create a combined dataframe of all scenarios
    pp_scenarios_dfs = [df for p, df in get_power_plant_scenarios(pp.id)]
    stacked_scenarios = pd.concat(pp_scenarios_dfs, axis=0)

    # downloader for all scenarios
    with l:
        st.download_button(
            f"Download {pp.name} power plant scenarios",
            encode_csv(stacked_scenarios),
            f"{pp.name}_scenarios.csv",
            "Download all scenarios for power plant type.",
            use_container_width=True,
            type="primary",
        )

    # selector for single scenario
    with r:
        s = st.selectbox(
            "Power Plant Scenario",
            list(range(len(pp_scenarios))),
            format_func=lambda x: pp_scenarios[x].name,
            help="Select a power plant scenario to view its data",
        )

    # select scenario data
    scenario_df = pp_scenarios_dfs[s]
    scenario = pp_scenarios[s]

    # downloader for single scenario
    with r:
        st.download_button(
            f"Download {pp.name} {scenario.name} scenario",
            encode_csv(scenario_df),
            f"{pp.name}_{scenario.name}_scenario.csv",
            "Download scenario data",
            use_container_width=True,
            type="primary",
        )

    with chart_container:
        fig = px.line(
            scenario_df.reset_index(),
            x="Timestamp",
            y="capacities",
            title=f"{pp.name} {scenario.name} Scenario",
            labels={"capacities": "Capacity [kW]"},
        )
        st.plotly_chart(fig, use_container_width=True)


def password_protect():
    if settings.env == "dev":
        return True
    if "password" not in st.session_state:
        st.session_state.password = None
    if st.session_state.password == settings.password:
        return True
    else:
        password = st.text_input("Password", type="password")
        st.session_state.password = password
        result = st.session_state.password == settings.password
        if result:
            st.experimental_rerun()
        return result


render_title()
logged_in = password_protect()
if logged_in:
    buildings_tab, scenarios_tab, power_plants_tab = st.tabs(
        ["Buildings", "Scenarios", "Power Plants"]
    )
    with buildings_tab:
        render_buildings()
    with scenarios_tab:
        render_building_scenarios()
    with power_plants_tab:
        render_power_plants()
