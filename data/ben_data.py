# --------------------------------------------------------
# Based from SeCo codebase
# https://github.com/ServiceNow/seasonal-contrast
# --------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url
import torchvision.transforms as T
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

QUANTILES = {
    'min_q': {
        'B01': 1.0, 'B02': 50.0, 'B03': 61.0, 'B04': 27.0,
        'B05': 16.0, 'B06': 1.0, 'B07': 5.0, 'B08': 1.0, 
        'B8A': 1.0, 'B09': 1.0, 'B11': 4.0, 'B12': 6.0},
    'max_q': {
        'B01': 2050.0, 'B02': 1862.0, 'B03': 2001.0, 'B04': 2332.0,
        'B05': 2656.0, 'B06': 4039.0, 'B07': 4830.0, 'B08': 5119.0,
        'B8A': 5092.0, 'B09': 4881.0, 'B11': 4035.0, 'B12': 3139.0}
    }

LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

class Bigearthnet(Dataset):
    url = 'http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz'
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]

    def __init__(self, root, split, bands=3, transform=None, target_transform=None, download=False, use_new_labels=True, img_size=128):
        self.root = Path(root)
        self.split = split
        self.bands = RGB_BANDS if bands==3 else ALL_BANDS
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels
        self.image_size = (img_size, img_size)

        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)

        bad_patches = set()
        for url in self.bad_patches:
            filename = Path(url).name
            with open(self.root / filename) as f:
                bad_patches.update(f.read().splitlines())

        self.samples = []
        with open(self.root / f'{self.split}.txt') as f:
            for patch_id in f.read().splitlines():
                if patch_id not in bad_patches:
                    self.samples.append(self.root / self.subdir / patch_id)

    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.name

        channels = []
        for b in self.bands:
            ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
            ch = normalize(ch, min_q=QUANTILES['min_q'][b], max_q=QUANTILES['max_q'][b])
            channels.append(T.functional.resize(torch.from_numpy(ch).unsqueeze(0), self.image_size, 
                                interpolation=T.InterpolationMode.BILINEAR, antialias=True))
        img = torch.cat(channels, dim=0)

        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.int_)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target

def scale_tensor(sample):
        """Transform a single sample from the Dataset."""
        sample = sample.float().div(255)
        return sample
    
def build_transform(config, split='train'):
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0 or config.AUG.CUTMIX_MINMAX is not None
    if split=='train' and mixup_active:
        transforms = T.Compose([
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            scale_tensor,
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    elif split=='train':
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            scale_tensor,
        ])
    else:
        transforms = T.Compose([
            scale_tensor,
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    return transforms

# Creating a subset of BigEarthNet like in SeCo and SatMAE
class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)