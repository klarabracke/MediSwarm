import os
import shutil
import pandas as pd
import numpy as np

image_dir = r"C:\Users\bracke\Documents\all"
excel_path = r"C:\Users\bracke\Documents\all_split\Clinical_and_Other_Features.xlsx"
output_base = r"C:\Users\bracke\Documents\all_split"

hospital_1 = os.path.join(output_base, "Krankenhaus_1")  # M2
hospital_2 = os.path.join(output_base, "Krankenhaus_2")  # M0.1
hospital_3 = os.path.join(output_base, "Krankenhaus_3")  # M0.2

# Erstelle Zielordner, wenn sie noch nicht existieren
for hospital in [hospital_1, hospital_2, hospital_3]:
    os.makedirs(hospital, exist_ok=True)

df = pd.read_excel(excel_path, header=1)  
df = df[["Patient ID", "Manufacturer", "Bilateral Information", "Tumor Location"]]
df.dropna(subset=["Patient ID", "Manufacturer"], inplace=True)

df["Patient ID"] = df["Patient ID"].astype(str).str.strip()  
df["Manufacturer"] = df["Manufacturer"].astype(int)

def normalize_bilateral(val):
    if pd.isna(val):
        return "NC"  
    if isinstance(val, str):
        return val.strip().upper()  
    if isinstance(val, (int, float)):
        return int(val)  
    return str(val).upper()  

df["Bilateral Information"] = df["Bilateral Information"].apply(normalize_bilateral)

df["Target Hospital"] = None

# Krankenhaus 1: Alle Patienten von Manufacturer 2
df.loc[df["Manufacturer"] == 2, "Target Hospital"] = hospital_1

# Beidseitige Krebsfälle für Krankenhaus 2 (Bilateral Information == 1)
bilateral_cancer_cases = df[(df["Bilateral Information"] == 1) & (df["Manufacturer"] == 0)]

# Einseitige Krebsfälle für Krankenhaus 2 (L-Krebs und R-Krebs)
left_cancer_cases = df[(df["Tumor Location"] == "L") & (df["Manufacturer"] == 0) & (df["Bilateral Information"] == 0)]
right_cancer_cases = df[(df["Tumor Location"] == "R") & (df["Manufacturer"] == 0) & (df["Bilateral Information"] == 0)]

n_left = len(left_cancer_cases)
n_right = len(right_cancer_cases)

left_cancer_split = left_cancer_cases.sample(n=n_left//2, random_state=42)  
right_cancer_split = right_cancer_cases.sample(n=int(np.ceil(n_right / 2)), random_state=42)  

#  Krankenhaus 2: Beidseitige Krebsfälle und zufällig ausgewählte L und R Krebsfälle
df.loc[df["Patient ID"].isin(bilateral_cancer_cases["Patient ID"]), "Target Hospital"] = hospital_2
df.loc[df["Patient ID"].isin(left_cancer_split["Patient ID"]), "Target Hospital"] = hospital_2
df.loc[df["Patient ID"].isin(right_cancer_split["Patient ID"]), "Target Hospital"] = hospital_2

# Krankenhaus 3: Patienten ohne Krebs und die andere Hälfte der einseitigen Krebsfälle
no_cancer = df[(df["Bilateral Information"] == "NC") & (df["Manufacturer"] == 0)]
remaining_left_cancer_cases = left_cancer_cases.drop(left_cancer_split.index)  
remaining_right_cancer_cases = right_cancer_cases.drop(right_cancer_split.index) 

df.loc[df["Patient ID"].isin(no_cancer["Patient ID"]), "Target Hospital"] = hospital_3
df.loc[df["Patient ID"].isin(remaining_left_cancer_cases["Patient ID"]), "Target Hospital"] = hospital_3
df.loc[df["Patient ID"].isin(remaining_right_cancer_cases["Patient ID"]), "Target Hospital"] = hospital_3

# Kopieren der Bilddateien unter Beibehaltung der Ordnerstruktur
for _, row in df.iterrows():
    patient_id = row["Patient ID"]
    target_dir = row["Target Hospital"]

    if pd.isna(target_dir):
        print(f"Kein Zielkrankenhaus für Patient {patient_id} definiert. Überspringe")
        continue

    matching_folders = [f for f in os.listdir(image_dir) if f.startswith(patient_id[-3:])]
    if not matching_folders:
        print(f"Kein Ordner für Patient {patient_id} gefunden. Überspringe")
        continue

    for folder_name in matching_folders:
        folder_path = os.path.join(image_dir, folder_name)
        src_file = os.path.join(folder_path, "sub.nii.gz")
        if not os.path.exists(src_file):
            print(f"Bild fehlt in {folder_path}. Überspringe")
            continue

        # Erstelle das Zielverzeichnis im Zielkrankenhaus (behalte die ursprüngliche Ordnerstruktur bei)
        patient_subfolder = os.path.join(target_dir, folder_name)
        os.makedirs(patient_subfolder, exist_ok=True)

        dest_file = os.path.join(patient_subfolder, "sub.nii.gz")
        
        if os.path.exists(dest_file):
            print(f"Bild {dest_file} existiert bereits. Überspringe")
            continue

        shutil.copy(src_file, dest_file)
        print(f" Kopiert: {src_file} → {dest_file}")

print("Verteilung abgeschlossen")

h1 = df[df["Target Hospital"] == hospital_1].count()
h2 = df[df["Target Hospital"] == hospital_2].count()
h3 = df[df["Target Hospital"] == hospital_3].count()
print(f"Target Hospital 1:{h1}")
print(f"Target Hospital 2:{h2}")
print(f"Target Hospital 3:{h3}")
