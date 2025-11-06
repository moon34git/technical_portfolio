import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torchvision

import os
import chardet
import torch
import pandas as pd
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as transforms

class NMCDataset(Dataset):
    def __init__(self, image_dir, csv_path, train_ratio=1.0, train=True, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        self.train_ratio = train_ratio

        result_df = pd.read_csv(csv_path)
        result_df = self._process_labels(result_df)

        full_image_paths = result_df['img_path'].tolist()
        labels = result_df['label'].tolist()

        if train_ratio < 1.0:
            train_imgs, test_imgs, train_labels, test_labels = train_test_split(
                full_image_paths,
                labels,
                train_size=train_ratio,
                stratify=labels,
                random_state=34
            )
            if train:
                selected_imgs = train_imgs
                selected_labels = train_labels
            else:
                selected_imgs = test_imgs
                selected_labels = test_labels
        else:
            selected_imgs = full_image_paths
            selected_labels = labels

        self.image_files = selected_imgs
        self.labels = selected_labels

    def _process_labels(self, df):
        def process_label(x):
            if isinstance(x, str):
                return [int(label) for label in x.split(',') if label]
            else:
                raise ValueError(f"Unexpected label value: {x}")

        df['label'] = df['label'].apply(process_label)

        exclude_labels = {'7', '8', '9', '10'}
        df['label'] = df['label'].apply(lambda x: [label for label in x if str(label) not in exclude_labels])

        df = df[df['label'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        df['label'] = df['label'].apply(lambda x: 0 if 0 in x else 1)

        df = df.rename(columns={df.columns[1]: 'img_path'})

        return df

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, torch.tensor(label, dtype=torch.float32)
    
class APTOSDataset(Dataset):
    def __init__(self, image_dir, csv_path, train_ratio=1.0, train=True, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[['id_code', 'label']].copy()
        df['id_code'] = df['id_code'].apply(lambda x: f"{x}.png" if not x.endswith('.png') else x)

        available_images = set(os.listdir(image_dir))
        df = df[df['id_code'].isin(available_images)].reset_index(drop=True)

        images = df['id_code'].tolist()
        labels = df['label'].tolist()

        if train_ratio < 1.0:
            train_imgs, test_imgs, train_labels, test_labels = train_test_split(
                images, labels,
                train_size=train_ratio,
                stratify=labels,
                random_state=34
            )
            self.image_files = train_imgs if train else test_imgs
            self.labels = train_labels if train else test_labels
        else:
            self.image_files = images
            self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)



class ODIRDataset(Dataset):
    def __init__(self, image_dir, csv_path, train_ratio=1.0, train=True, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[['images', 'label']].copy()

        available_images = set(os.listdir(image_dir))
        df = df[df['images'].isin(available_images)].reset_index(drop=True)

        images = df['images'].tolist()
        labels = df['label'].tolist()

        if train_ratio < 1.0:
            train_imgs, test_imgs, train_labels, test_labels = train_test_split(
                images, labels,
                train_size=train_ratio,
                stratify=labels,
                random_state=34
            )
            self.image_files = train_imgs if train else test_imgs
            self.labels = train_labels if train else test_labels
        else:
            self.image_files = images
            self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)


class IDRiDDataset(Dataset):
    def __init__(self, image_dir, csv_path, train_ratio=1.0, train=True, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[['Image name', 'label']].copy()

        available_images = set(os.listdir(image_dir))
        df = df[df['Image name'].isin(available_images)].reset_index(drop=True)

        images = df['Image name'].tolist()
        labels = df['label'].tolist()

        if train_ratio < 1.0:
            train_imgs, test_imgs, train_labels, test_labels = train_test_split(
                images, labels,
                train_size=train_ratio,
                stratify=labels,
                random_state=34
            )
            self.image_files = train_imgs if train else test_imgs
            self.labels = train_labels if train else test_labels
        else:
            self.image_files = images
            self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)


class MESSIDORDataset(Dataset):
    def __init__(self, image_dir, csv_path, train_ratio=1.0, train=True, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[['Image name', 'label']].copy()

        available_images = set(os.listdir(image_dir))
        df = df[df['Image name'].isin(available_images)].reset_index(drop=True)

        images = df['Image name'].tolist()
        labels = df['label'].tolist()

        if train_ratio < 1.0:
            train_imgs, test_imgs, train_labels, test_labels = train_test_split(
                images, labels,
                train_size=train_ratio,
                stratify=labels,
                random_state=34
            )
            self.image_files = train_imgs if train else test_imgs
            self.labels = train_labels if train else test_labels
        else:
            self.image_files = images
            self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)

def get_partial_labeled_subset(dataset, label_ratio=0.8, random_state=42):
    """
    Return a Subset with only a portion of samples assumed to have labels.
    - label_ratio: Ratio of samples assumed to have labels (0.8 -> 80%)
    - random_state: Random seed for reproducibility
    """
    labels = []
    print("[DEBUG] Starting get_partial_labeled_subset")
    print(f"[DEBUG] Total samples: {len(dataset)}")
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(f"[DEBUG] Collecting labels... {i}/{len(dataset)}")
        try:
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            labels.append(label)
        except Exception as e:
            raise RuntimeError(f"Failed to extract label from sample {i}: {e}")
    print(f"[DEBUG] Label collection complete: Number of classes = {len(np.unique(labels))}")
    labels = np.array(labels)
    labeled_indices = []
    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        n_labeled = int(len(class_indices) * label_ratio)
        print(f"[DEBUG] Class {class_id}: Selecting {n_labeled} out of {len(class_indices)}")
        rng = np.random.default_rng(seed=random_state)
        selected = rng.choice(class_indices, n_labeled, replace=False)
        labeled_indices.extend(selected)
    print(f"[DEBUG] Total selected labeled samples: {len(labeled_indices)}")
    return Subset(dataset, labeled_indices)
