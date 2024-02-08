import pandas as pd
import streamlit as st
from lib.client import client
from lib.supa import Building

st.set_page_config(layout="wide", page_title="MIT Decarbonization")

@st.cache_data
def get_all_buildings():
    df = pd.DataFrame(client.table("Building").select("*").execute().data).set_index('id')
    # create csv bytes
    csv = df.to_csv().encode()
    return df, csv

def render_title():
    st.title("MIT Decarbonization")

def render_buildings():
    st.header("Buildings")
    all_buildings, all_buildings_csv = get_all_buildings()
    st.download_button("Download all buildings", all_buildings_csv, "buildings_metadata.csv", "Download all buildings", use_container_width=True, type="primary")
    building_id = st.selectbox('Building', all_buildings.index, format_func=lambda x: all_buildings.loc[x, 'name'], help='Select a building to view its data')
    building = all_buildings.loc[building_id]
    st.dataframe(building)


render_title()
st.divider()
render_buildings()




