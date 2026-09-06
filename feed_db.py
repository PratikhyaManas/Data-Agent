"""
Seed the SQLite database from the sample CSVs in data/.
Run once before using the SQL Analyst agent: `python feed_db.py`
"""
import os
import sqlite3

import pandas as pd
from dotenv import load_dotenv

from project_bootstrap import ensure_repo_root

ensure_repo_root()
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/data_agent.db")
CSV_TABLE_MAP = {
    "data/vehicles.csv": "vehicles",
    "data/users.csv": "users",
}


def seed():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    for csv_path, table_name in CSV_TABLE_MAP.items():
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_path} (not found)")
            continue
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {csv_path} -> table `{table_name}` ({len(df)} rows)")
    conn.close()
    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    seed()
