import shutil

import pandas as pd
import os

df = pd.read_excel("Dataset/CASME2-coding-20140508.xlsx",dtype={"Subject": str})
dir_path = "Dataset/CASME2_retinaface/"
out_dir_path = "Dataset/CASME2_onset_apex_offset_retinaface/"

def modify(old, new):
    if os.path.exists(old):
        os.rename(old, new)

for index, row in df.iterrows():

    print(row["Filename"])
    print(row.Filename)

    # os.makedirs(, exist_ok=True)

    # for col in df.columns:
    #     if col in ["OnsetFrame", "ApexFrame", "OffsetFrame"] and col:
    #         old_path = f"{dir_path}sub{row['Subject']}/{row['Filename']}/img{row[col]}.jpg"
    #         if col == "OnsetFrame":
    #             new_path = f"{dir_path}sub{row['Subject']}/{row['Filename']}/img{row[col]}_onset.jpg"
    #             modify(old_path, new_path)
    #         elif col == "ApexFrame":
    #             new_path = f"{dir_path}sub{row['Subject']}/{row['Filename']}/img{row[col]}_apex.jpg"
    #             modify(old_path, new_path)
    #         elif col == "OffsetFrame":
    #             new_path = f"{dir_path}sub{row['Subject']}/{row['Filename']}/img{row[col]}_offset.jpg"
    #             modify(old_path, new_path)