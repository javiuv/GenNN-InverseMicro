from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CleanImageDataset(Dataset):
    def __init__(self, root_dir, image_size=96):
        """
        Dataset for loading clean images only.
        root_dir example:
            .../processed_denoising_final/train/clean
        """
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(list(self.root_dir.glob("*.png")))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image