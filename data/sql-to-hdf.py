import os
from pathlib import Path

import pandas as pd
from archetypal.idfclass.sql import Sql
from tqdm.auto import tqdm

if __name__ == "__main__":
    results_path = Path("data") / "baseline_results_agg.hdf"
    if results_path.exists():
        os.remove(results_path)
    files = (Path("data") / "results").glob("*.sql")
    for file in tqdm(files):
        sql = Sql(file)
        df = sql.timeseries_by_name(
            [
                "Zone Electric Equipment Electricity Energy",
                "Zone Lights Electricity Energy",
                "Zone Ideal Loads Supply Air Total Heating Energy",
                "Zone Ideal Loads Supply Air Total Cooling Energy",
                "Water Use Equipment Heating Energy",
            ],
            "Hourly",
        )
        df = (
            df.T.groupby(["IndexGroup", "Name"])
            .sum()
            .T.droplevel(0, axis=1)
            .rename(
                columns={
                    "Zone Ideal Loads Supply Air Total Cooling Energy": "cooling",
                    "Zone Ideal Loads Supply Air Total Heating Energy": "heating",
                    "Zone Electric Equipment Electricity Energy": "equipment",
                    "Zone Lights Electricity Energy": "lighting",
                    "Water Use Equipment Heating Energy": "water",
                }
            )
        )
        df.columns.name = "End Use"
        df.to_hdf(results_path, key=file.stem, mode="a")
