import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Optional: Dummy-Modell, falls du keine Modellklasse laden willst
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Für PyTorch 2.6+ notwendig, wenn du eigene Optimizer geladen hast
torch.serialization.add_safe_globals([torch.optim.AdamW])  # Korrektur hier

def load_checkpoint(checkpoint_path):
    try:
        # Sicherstellen, dass der Checkpoint auf der CPU geladen wird
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints von {checkpoint_path}: {e}")
        return None

def compare_checkpoints(checkpoints):
    results = []
    for hospital, path in checkpoints.items():
        print(f"Lade Checkpoint für {hospital} ...")
        checkpoint = load_checkpoint(path)
        if checkpoint is None:
            results.append((hospital, 0.0))
            continue

        # Ausgeben, was im Checkpoint gespeichert ist
        print(f"Inhalt des Checkpoints für {hospital}:")
        print(checkpoint.keys())  # Zeigt alle Schlüssel im Checkpoint

        # Beispielhafte Metrik auslesen (falls gespeichert)
        auroc = checkpoint.get("val/auroc", 0.0)
        print(f"AUROC für {hospital}: {auroc}")  # Ausgabe des AUROC-Werts

        results.append((hospital, auroc))

    return results

def plot_comparison(results):
    hospitals = [r[0] for r in results]
    scores = [r[1] for r in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(hospitals, scores, color='steelblue')
    plt.title("Vergleich der AUROC-Werte (val/auroc)")
    plt.ylabel("AUROC")
    plt.ylim(0, 1)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{score:.2f}", ha='center')

    plt.tight_layout()
    plt.show()

#  Korrigierte Pfade
checkpoints = {
    "Krankenhaus_1": r"C:\Users\bracke\Documents\all_split\scratch_K_1\2025_04_24_144120_DUKE_ResNet50_swarm_learning\last.ckpt",
    "Krankenhaus_2": r"C:\Users\bracke\Documents\all_split\scratch_K_2\2025_04_24_144119_DUKE_ResNet50_swarm_learning\last.ckpt",
    "Krankenhaus_3": r"C:\Users\bracke\Documents\all_split\scratch_K_3\2025_04_24_144119_DUKE_ResNet50_swarm_learning\last.ckpt"
}

# Ausführen
results = compare_checkpoints(checkpoints)
plot_comparison(results)
