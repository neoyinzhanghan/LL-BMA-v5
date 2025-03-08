import pandas as pd
import sqlite3


def get_path_data(case_numbers):
    conn = sqlite3.connect("/media/hdd3/neo/copath.db")
    placeholders = ",".join(["?"] * len(case_numbers))
    query = f"SELECT specnum_formatted, part_description, text_data_final FROM heme_v2 WHERE specnum_formatted IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=case_numbers)
    df = df.drop_duplicates(subset="specnum_formatted", keep="first")
    df.reset_index(drop=True, inplace=True)
    conn.close()
    return df


def lab_value_query(case_numbers, columns, conn):
    placeholders = ",".join(["?"] * len(case_numbers))
    cols = ", ".join(columns)
    cols = ", ".join([f'"{col}"' for col in columns])
    query = f"SELECT specnum_formatted, {cols} FROM heme_v2 WHERE specnum_formatted IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=case_numbers)
    return df


def merge_df(main_df, new_df, columns):
    merged_df = pd.merge(main_df, new_df, on="specnum_formatted", how="left")
    for col in columns:
        if col + "_y" in merged_df.columns:
            merged_df[col] = merged_df[col + "_y"]
            merged_df.drop([col + "_x", col + "_y"], axis=1, inplace=True)
    return merged_df


def get_diff(df):
    diff_cols = [
        "blasts",
        "blast-equivalents",
        "promyelocytes",
        "myelocytes",
        "metamyelocytes",
        "neutrophils/bands",
        "monocytes",
        "eosinophils",
        "erythroid precursors",
        "lymphocytes",
        "plasma cells",
    ]
    conn = sqlite3.connect("/media/ssd2/clinical_text_data/Copath Database/copath.db")
    new_df = lab_value_query(df["specnum_formatted"].unique(), diff_cols, conn)
    conn.close()
    new_df = new_df.drop_duplicates(subset="specnum_formatted", keep="first")
    updated_df = merge_df(df, new_df, diff_cols)
    return updated_df


def get_cbc(df):
    cbc_cols = [
        "WBC",
        "RBC",
        "HGB",
        "HCT",
        "MCV",
        "MCH",
        "MCHC",
        "RDW",
        "Platelets",
        "Blast",
        "Neutrophil",
        "Mono",
        "Eos",
        "Baso",
        "Immature Granulocyte",
        "Lymph",
        "Plasma Cell",
        "Nucleated RBC",
        "Abs Neut",
        "Abs Mono",
        "Absolute Eosinophil",
        "Absolute Basophil",
        "Absolute Immature",
        "Abs Lymph",
        "Abnormal Lymph.*",
        "Hairy Cells",
        "Other.*",
    ]
    conn = sqlite3.connect("/media/ssd2/clinical_text_data/Copath Database/copath.db")
    new_df = lab_value_query(df["specnum_formatted"].unique(), cbc_cols, conn)
    new_df = new_df.drop_duplicates(subset="specnum_formatted", keep="first")
    conn.close()
    updated_df = merge_df(df, new_df, cbc_cols)
    return updated_df


if __name__ == "__main__":
    case_numbers = ["your_accession_number_here"]
    df = get_path_data(case_numbers)
    print(df.head())

    # save the data to a csv file
    df.to_csv("/media/hdd3/neo/test_path_data.csv", index=False)

    # get the differential data
    diff_df = get_diff(df)

    diff_df.to_csv("/media/hdd3/neo/test_diff_data.csv", index=False)
