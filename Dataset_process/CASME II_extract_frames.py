import pandas as pd
import shutil
import os

df = pd.read_excel("..Dataset/CASME2-coding-20140508.xlsx",dtype={"Subject": str})
dir_path = "../Dataset/CASME2_retinaface/"
out_dir_path = "../Dataset/CASME2_onset_apex_offset_retinaface/"

lst = []

def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

for index, row in df.iterrows():

    folder_path = f"{dir_path}sub{row['Subject']}/{row['Filename']}"
    Onset_path = f"{folder_path}/img{row['OnsetFrame']}.jpg"
    Offset_path = f"{folder_path}/img{row['OffsetFrame']}.jpg"
    Apex_path = f"{folder_path}/img{row['ApexFrame']}.jpg"

    if os.path.exists(Onset_path) and os.path.exists(Offset_path) and os.path.exists(Apex_path):  #确定三个帧都有

        out_path = f"{out_dir_path}sub{row['Subject']}"
        os.makedirs(out_path, exist_ok=True)

        for col in df.columns:
            if col in ["OnsetFrame", "ApexFrame", "OffsetFrame"]:
                old_path = f"{dir_path}sub{row['Subject']}/{row['Filename']}/img{row[col]}.jpg"
                if col == "OnsetFrame":
                    new_path = f"{out_path}/{row['Filename']}_onset.jpg"
                    shutil.copy(old_path, new_path)
                elif col == "ApexFrame":
                    new_path = f"{out_path}/{row['Filename']}_apex.jpg"
                    shutil.copy(old_path, new_path)
                elif col == "OffsetFrame":
                    new_path = f"{out_path}/{row['Filename']}_offset.jpg"
                    shutil.copy(old_path, new_path)
    else:
        lst.append(index)
        print(lst)