from lightning.pytorch import LightningDataModule, LightningModule
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
import torch
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms


def base64_to_tensor(base64_str, img_size=96):
    image = Image.open(BytesIO(base64.b64decode(base64_str)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts to tensor with shape [C, H, W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)  # No unsqueeze needed!


class ImageCaptionDataset(Dataset):
    def __init__(self, df, encode_fn, stoi, img_size=96, transform=None):
        self.df = df
        self.img_size = img_size
        self.transform = transform
        self.encode = encode_fn
        self.stoi = stoi

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = base64_to_tensor(row["b64string_images"], self.img_size)
        caption = row["caption"]
        encoded_caption = torch.tensor(self.encode(caption), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, encoded_caption

    def collate_fn(self, batch):
        images, captions = zip(*batch)

        images = torch.stack(images, dim=0)

        # Pad captions
        max_length = max(len(cap) for cap in captions)
        padded_captions = torch.full(
            (len(captions), max_length), fill_value=self.stoi["<pad>"], dtype=torch.long
        )
        for i, cap in enumerate(captions):
            padded_captions[i, : len(cap)] = cap

        # Create targets
        targets = torch.cat(
            [
                padded_captions[:, 1:],
                torch.full(
                    (len(captions), 1),
                    fill_value=self.stoi["<pad>"],
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )

        # Ensure targets and padded_captions have the same dimensions.
        if targets.size(1) > padded_captions.size(1):
            targets = targets[:, : padded_captions.size(1)]
        elif targets.size(1) < padded_captions.size(1):
            targets = torch.cat(
                [
                    targets,
                    torch.full(
                        (len(captions), padded_captions.size(1) - targets.size(1)),
                        fill_value=self.stoi["<pad>"],
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )

        return images, padded_captions, targets


class VLMDataModule(LightningDataModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(
        self,
        encode_fn,
        stoi,
        input_path: str = "../images/inputs.csv",
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.img_size = 96
        self.batch_size = batch_size
        self.val_batch_size = 8
        self.num_workers = 4
        self.encode_fn = encode_fn
        self.stoi = stoi

    def setup(self, stage: str) -> None:
        current_dir = os.path.dirname(__file__)
        filename = os.path.join(current_dir, self.input_path)
        df = pd.read_csv(filename)
        df = pd.concat([df] * 30)[["b64string_images", "caption"]]
        n = int(0.9 * len(df))
        df_train = df.iloc[:n]
        df_val = df.iloc[n:]

        self.train_dataset = ImageCaptionDataset(
            df_train, self.encode_fn, self.stoi, img_size=self.img_size
        )
        self.val_dataset = ImageCaptionDataset(
            df_val, self.encode_fn, self.stoi, img_size=self.img_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,  # Typically no need to shuffle validation data
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
        )
