import os
import shutil
import pandas as pd

image_dir = r"C:\Users\bracke\Documents\all"
excel_path = r"C:\Users\bracke\Documents\all_split\Clinical_and_Other_Features.xlsx"
output_base = r"C:\Users\bracke\Documents\all_split"

hospital_1 = os.path.join(output_base, "Krankenhaus_1")  # Manufacturer 2
hospital_2 = os.path.join(output_base, "Krankenhaus_2")  # Manufacturer 0 & 3T
hospital_3 = os.path.join(output_base, "Krankenhaus_3")  # Manufacturer 0 & 1T

for hospital in [hospital_1, hospital_2, hospital_3]:
    os.makedirs(hospital, exist_ok=True)

# Excel laden
df = pd.read_excel(excel_path, header=1)
df = df[["Patient ID", "Manufacturer", "Field Strength (Tesla)"]]
df.dropna(subset=["Patient ID", "Manufacturer", "Field Strength (Tesla)"], inplace=True)

df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
df["Manufacturer"] = df["Manufacturer"].astype(int)
df["Field Strength (Tesla)"] = df["Field Strength (Tesla)"].astype(float)

df["Target Hospital"] = None

# Krankenhaus-Zuweisung
df.loc[df["Manufacturer"] == 2, "Target Hospital"] = hospital_1
df.loc[(df["Manufacturer"] == 0) & (df["Field Strength (Tesla)"] == 3), "Target Hospital"] = hospital_2
df.loc[(df["Manufacturer"] == 0) & (df["Field Strength (Tesla)"] == 1), "Target Hospital"] = hospital_3

# Bilddateien kopieren
for _, row in df.iterrows():
    patient_id = row["Patient ID"]
    target_dir = row["Target Hospital"]

    if pd.isna(target_dir):
        print(f"Kein Zielkrankenhaus für Patient {patient_id} definiert. Überspringe.")
        continue

    matching_folders = [f for f in os.listdir(image_dir) if f.startswith(patient_id[-3:])]
    if not matching_folders:
        print(f"Kein Ordner für Patient {patient_id} gefunden. Überspringe.")
        continue

    for folder_name in matching_folders:
        folder_path = os.path.join(image_dir, folder_name)
        src_file = os.path.join(folder_path, "sub.nii.gz")
        if not os.path.exists(src_file):
            print(f"Bild fehlt in {folder_path}. Überspringe.")
            continue

        patient_subfolder = os.path.join(target_dir, folder_name)
        os.makedirs(patient_subfolder, exist_ok=True)

        dest_file = os.path.join(patient_subfolder, "sub.nii.gz")
        if os.path.exists(dest_file):
            print(f"Bild {dest_file} existiert bereits. Überspringe.")
            continue

        shutil.copy(src_file, dest_file)
        print(f"Kopiert: {src_file} → {dest_file}")


h1 = df[df["Target Hospital"] == hospital_1].shape[0]
h2 = df[df["Target Hospital"] == hospital_2].shape[0]
h3 = df[df["Target Hospital"] == hospital_3].shape[0]
print(f"Krankenhaus 1 (Manufacturer 2): {h1} Patienten")
print(f"Krankenhaus 2 (Manufacturer 0 & 3T): {h2} Patienten")
print(f"Krankenhaus 3 (Manufacturer 0 & 1T): {h3} Patienten")
