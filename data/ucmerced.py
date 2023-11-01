# --------------------------------------------------------
# Based from TorchGeo codebase
# https://github.com/microsoft/torchgeo
# --------------------------------------------------------

"""UC Merced dataset."""
import os
import torch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from torchgeo.datasets.geo import NonGeoClassificationDataset
from torchvision.datasets import ImageFolder
from torchgeo.datasets.utils import check_integrity, download_url, extract_archive

import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp

class UCMerced(NonGeoClassificationDataset):
    """UC Merced dataset.

    The `UC Merced <http://weegee.vision.ucmerced.edu/datasets/landuse.html>`__
    dataset is a land use classification dataset of 2.1k 256x256 1ft resolution RGB
    images of urban locations around the U.S. extracted from the USGS National Map Urban
    Area Imagery collection with 21 land use classes (100 images per class).

    Dataset features:

    * land use class labels from around the U.S.
    * three spectral bands - RGB
    * 21 classes

    Dataset classes:

    * agricultural
    * airplane
    * baseballdiamond
    * beach
    * buildings
    * chaparral
    * denseresidential
    * forest
    * freeway
    * golfcourse
    * harbor
    * intersection
    * mediumresidential
    * mobilehomepark
    * overpass
    * parkinglot
    * river
    * runway
    * sparseresidential
    * storagetanks
    * tenniscourt

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://dl.acm.org/doi/10.1145/1869790.1869829
    """

    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"  # 318 MB
    filename = "UCMerced_LandUse.zip"
    md5 = "5b7ec56793786b6dc8a908e8854ac0e4"

    base_dir = os.path.join("UCMerced_LandUse", "Images")
    classes = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "f2fb12eb2210cfb53f93f063a35ff374",
        "val": "11ecabfc52782e5ea6a9c7c0d263aca0",
        "test": "046aff88472d8fc07c4678d03749e28d",
    }

    def __init__(self, root="data", split="train", transform=None, download=False, checksum=False, image_size=256):
        """Initialize a new UC Merced dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transform: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in self.splits
        self.root = root
        self.transforms = transform
        self.download = download
        self.checksum = checksum
        self.image_size = image_size
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"uc_merced-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transform,
            is_valid_file=is_in_split,
        )

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.
        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )
        return integrity

    def _verify(self) -> None:
        """Verify the integrity of the dataset.
        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.base_dir)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        if self._check_integrity():
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f"uc_merced-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)


    def __getitem__(self, index: int):
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int):
        """Load a single image and it's class label.
        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        return img, label
    
def build_transform(config, split='train'):
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0 or config.AUG.CUTMIX_MINMAX is not None
    if split=='train' and mixup_active:
        transforms = T.Compose([
                T.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE), interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    elif split=='train':
        transforms = T.Compose([
                T.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE), interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    else:
        transforms = T.Compose([
                T.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE), interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    return transforms