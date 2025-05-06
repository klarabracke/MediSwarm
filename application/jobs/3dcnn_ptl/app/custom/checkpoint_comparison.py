import monai.networks.nets as nets
import torch
import nibabel as nib
import numpy as np


class ResNet503DFeatureExtractor:
    def __init__(
        self,
        n_input_channels: int = 1,
        num_classes: int = 1,
        block: str = "basic",
        layers: list = [3, 4, 6, 3],
        block_inplanes: list = [64, 128, 256, 512],
        checkpoint_path: str = None,
        feature_extractor: bool = True,
    ):
        self.feature_extractor = nets.ResNet(
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            n_input_channels=n_input_channels,
            num_classes=num_classes,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        if feature_extractor:
            self.remove_classification_head()
        self.feature_extractor.to(self.device)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        checkpoint = checkpoint["state_dict"]

        new_checkpoint = {}
        for k, v in checkpoint.items():
            new_key = k.replace("model.", "")
            new_checkpoint[new_key] = v

        self.feature_extractor.load_state_dict(new_checkpoint)
        print(f"Checkpoint loaded successfully from {checkpoint_path}")

    def remove_classification_head(self):
        self.feature_extractor.fc = torch.nn.Identity()

    def forward(self, x):
        x = x.to(self.device)
        extracted_features = self.feature_extractor(x)
        flattened_features = extracted_features.view(extracted_features.size(0), -1)
        return flattened_features

    def predict(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            prediction_features = self.forward(x)
        return prediction_features


def load_nifti_image_as_tensor(path, device):
    img = nib.load(path).get_fdata()
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=0)  
    img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
    return img_tensor



checkpoint_paths = {
    "Krankenhaus_1": r"C:\Users\bracke\Documents\all_split\scratch_K_1\2025_04_24_144120_DUKE_ResNet50_swarm_learning\last.ckpt",
    "Krankenhaus_2": r"C:\Users\bracke\Documents\all_split\scratch_K_2\2025_04_24_144119_DUKE_ResNet50_swarm_learning\last.ckpt",
    "Krankenhaus_3": r"C:\Users\bracke\Documents\all_split\scratch_K_3\2025_04_24_144119_DUKE_ResNet50_swarm_learning\last.ckpt"
}



image_path = r"C:/Users/bracke/Documents/all_split/Krankenhaus_1/001_left/sub.nii.gz"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = load_nifti_image_as_tensor(image_path, device)


features = {}
for name, path in checkpoint_paths.items():
    print(f"\nVerarbeite {name} ...")
    model = ResNet503DFeatureExtractor(checkpoint_path=path)
    feat = model.predict(input_tensor)
    features[name] = feat.cpu().numpy()
    print(f"{name} Feature-Vektor-Shape: {feat.shape}")


from sklearn.metrics.pairwise import cosine_similarity

print("\n Cosine Similarities:")
names = list(features.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        sim = cosine_similarity(features[names[i]], features[names[j]])
        print(f"{names[i]} vs {names[j]}: {sim[0, 0]:.4f}")
