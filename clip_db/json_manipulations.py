import pandas as pd
import json


def upload_new_json(json_data, clip_id):
    """
    Convert Json Data to SQL DB per Type.
    Every SQL table will be recognized by CLIP_id and the type name.
    Also the original json should be stored in no_sql db, holds the raw data of the jsons, clip_id is the key.

    Args:
        json_data (_type_): _description_
    """
    # Convert JSON to DataFrame
    df = pd.DataFrame(json.loads(json_data))

    # Split the DataFrame based on the 'type' attribute
    # Drop NaN to get DataFrames by specific type
    tables = {t: df[df['type'] == t].drop(columns='type') for t in df['type'].dropna().unique()}

    # Handle elements without 'type' (NaN in the 'type' column)
    if df['type'].isna().any():
        tables['no_type'] = df[df['type'].isna()]
        # Create tables in the SQL database for each type
    for type_name, table_df in tables.items():
        # Create or replace table for each type in SQL
        table_name = f"{clip_id}_{type_name}"
        table_df.to_sql(table_name, if_exists="replace", index=False)
        print(f"Table '{table_name}' created in SQL DB.")


def update_cloud(DB):
    """
    Upload snapshot of all DB's related to the CLIP, the CLIP's are stored as CLIP with CLIP id.
    """
    print("Uploading snapshot of all SQL tables to the cloud...")


def ETL(CLIP_id, sql_query):
    """
    Give the user full interface to change values by SQL query.
    The logic is: The user should be able to change, delete, retrieve values by sql query.
    1) Find all the objects related to the query.
    2) Activate the required operation - in case of change of type: move the value to the proper table, remove if table
    is empty, create if table is new - update the type_to_attributes table.
    """


def edit(CLIP_id, json_data):
    """
    Given edited json of specific CLIP_id.
    Make a diff of all the changed objects - get a diff json.
    Find all the objects requires edit by their type via the old json.
    Change the objects values via the same logic of ETL section 2.
    """