import os
import shutil
import pandas as pd
import random

# Konfiguration 
random.seed(42)
test_split_percentage = 0.2

image_dir = r"C:\Users\bracke\Documents\all"
output_base = r"C:\Users\bracke\Documents\all_split"

df_path = r"C:\Users\bracke\Documents\df_mit_hospital.csv"
df = pd.read_csv(df_path)
df["Patient ID"] = df["Patient ID"].astype(str).str.strip()

# Krankenhaus-Mapping
hospital_1 = os.path.join(output_base, "Krankenhaus_1")
hospital_2 = os.path.join(output_base, "Krankenhaus_2")
hospital_3 = os.path.join(output_base, "Krankenhaus_3")

test_hospital_1 = os.path.join(output_base, "Test_1")
test_hospital_2 = os.path.join(output_base, "Test_2")
test_hospital_3 = os.path.join(output_base, "Test_3")

# Zielordner für Testdaten anlegen
for test_dir in [test_hospital_1, test_hospital_2, test_hospital_3]:
    os.makedirs(test_dir, exist_ok=True)

# Funktion zum Splitten
def split_test_data(df_hospital, percentage):
    test_size = int(len(df_hospital) * percentage)
    test_data = df_hospital.sample(n=test_size, random_state=42)
    return test_data

# Split pro Krankenhaus
test_1 = split_test_data(df[df["Target Hospital"] == hospital_1], test_split_percentage)
test_2 = split_test_data(df[df["Target Hospital"] == hospital_2], test_split_percentage)
test_3 = split_test_data(df[df["Target Hospital"] == hospital_3], test_split_percentage)

test_df = pd.concat([test_1, test_2, test_3])

# Bilder kopieren
for _, row in test_df.iterrows():
    patient_id = row["Patient ID"]
    target_hospital = row["Target Hospital"]

    if target_hospital == hospital_1:
        dest_dir = test_hospital_1
    elif target_hospital == hospital_2:
        dest_dir = test_hospital_2
    elif target_hospital == hospital_3:
        dest_dir = test_hospital_3
    else:
        print(f"Kein Zielkrankenhaus für Patient {patient_id}")
        continue

    matching_folders = [f for f in os.listdir(image_dir) if f.startswith(patient_id)]
    if not matching_folders:
        print(f"Kein Ordner für Patient {patient_id} gefunden. Überspringe")
        continue

    for folder_name in matching_folders:
        src_file = os.path.join(image_dir, folder_name, "sub.nii.gz")
        if not os.path.exists(src_file):
            print(f"Bild fehlt in {src_file}. Überspringe")
            continue

        dest_file = os.path.join(dest_dir, f"{folder_name}.nii.gz")
        if os.path.exists(dest_file):
            print(f"Bild {dest_file} existiert bereits. Überspringe")
            continue

        shutil.copy(src_file, dest_file)
        print(f"Kopiert: {src_file} → {dest_file}")

# Test-Patient:innen exportieren
test_df.to_csv(r"C:\Users\bracke\Documents\df_test.csv", index=False)

# Trainings-Patient:innen ableiten und speichern
train_df = df[~df["Patient ID"].isin(test_df["Patient ID"])]
train_df.to_csv(r"C:\Users\bracke\Documents\df_train.csv", index=False)

# Sanity Check
overlap = set(test_df["Patient ID"]).intersection(train_df["Patient ID"])
assert len(overlap) == 0, f"Doppelte Patient:innen im Train-Test-Split: {overlap}"
print("Kein Overlap zwischen Trainings- und Testdaten")

# Übersicht über den Testdatensatz
print("\nTestdatensatz-Übersicht:")
print(f"Krankenhaus 1 (Test_1): {len(test_1)} Patient:innen")
print(f"Krankenhaus 2 (Test_2): {len(test_2)} Patient:innen")
print(f"Krankenhaus 3 (Test_3): {len(test_3)} Patient:innen")
print(f"Gesamt: {len(test_df)} Test-Patient:innen\n")
