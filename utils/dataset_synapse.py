import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from tqdm import tqdm


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """Shared augmentation/resize transform, used by both Synapse and ACDC."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


# ==============================================================================
# Synapse
# ==============================================================================
# On-disk layout: train slices at {base_dir}/{case_name}.npz (keys 'image'/
# 'label'); validation/test volumes at {base_dir}/{vol_name}.npy.h5 (same
# keys). nclass==9 triggers Synapse's 13-organ -> 9-class label merge.

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        if self.nclass == 9:
            label[label == 5] = 0
            label[label == 9] = 0
            label[label == 10] = 0
            label[label == 12] = 0
            label[label == 13] = 0
            label[label == 11] = 5

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class Synapse_preloaded_dataset(Dataset):
    """Same as Synapse_dataset, but reads every sample into RAM once in
    __init__ so __getitem__ is a pure in-memory lookup (no per-epoch disk
    I/O)."""
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform
        self.split = split  # FIX: previously assigned to a local `split`
        # variable instead of `self.split`, so __getitem__'s
        # `self.split == "train"` check below would raise AttributeError the
        # moment a transform was supplied (i.e. on every training batch).
        sample_list = open(os.path.join(list_dir, split + '.txt')).readlines()
        data_dir = base_dir
        self.nclass = nclass

        self.data_cache = []
        for line in tqdm(sample_list, desc=f'Preloading Synapse {split} data', unit='sample'):
            case_name = line.strip('\n')
            if split == "train":
                data_path = os.path.join(data_dir, case_name + '.npz')
                data = np.load(data_path)
                image, label = data['image'], data['label']
            else:
                filepath = data_dir + "/{}.npy.h5".format(case_name)
                with h5py.File(filepath, 'r') as data:
                    image, label = data['image'][:], data['label'][:]

            # copy out of any memory-mapped / soon-to-be-closed file handle
            image = np.array(image)
            label = np.array(label)

            if nclass == 9:
                label[label == 5] = 0
                label[label == 9] = 0
                label[label == 10] = 0
                label[label == 12] = 0
                label[label == 13] = 0
                label[label == 11] = 5

            self.data_cache.append({
                'image': image,
                'label': label,
                'case_name': case_name,
            })

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        cached = self.data_cache[idx]
        sample = {'image': cached['image'], 'label': cached['label']}

        if self.transform and self.split == "train":
            sample = self.transform(sample)

        sample['case_name'] = cached['case_name']
        return sample


# ==============================================================================
# ACDC
# ==============================================================================
# On-disk layout differs from Synapse: train/valid slices live at
# {base_dir}/{split}/{slice_name} (note: nested under a per-split
# subdirectory, and slice_name from the list file is expected to already
# include the .npz extension), with keys 'img'/'label' (not 'image'/'label').
# Any other split value (e.g. a held-out volume split such as "test") is read
# directly from {base_dir}/{vol_name}. ACDC's standard 4-class label set
# (background, RV, myocardium, LV) needs no remapping, so there's no
# nclass-driven special case here the way there is for Synapse.
#
# NOTE: verify the exact split name your ACDC list files use for full-volume
# validation/test data (e.g. "test" vs "test_vol") and pass that split name
# in when constructing these classes from trainer.py — the else-branch below
# is written to be split-name-agnostic (anything that isn't "train"/"valid"
# falls into it), so it works with whatever name you use as long as it's
# consistent with your list_dir's .txt filename and the data's own layout.

class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, vol_name)
            data = np.load(filepath)
            image, label = data['img'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class ACDC_preloaded_dataset(Dataset):
    """Same RAM-preloading optimization as Synapse_preloaded_dataset,
    adapted to ACDC's file layout and 'img'/'label' npz keys."""
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        sample_list = open(os.path.join(list_dir, split + '.txt')).readlines()
        data_dir = base_dir

        self.data_cache = []
        for line in tqdm(sample_list, desc=f'Preloading ACDC {split} data', unit='sample'):
            case_name = line.strip('\n')
            if split == "train" or split == "valid":
                data_path = os.path.join(data_dir, split, case_name)
            else:
                data_path = os.path.join(data_dir, case_name)

            data = np.load(data_path)
            image = np.array(data['img'])
            label = np.array(data['label'])

            self.data_cache.append({
                'image': image,
                'label': label,
                'case_name': case_name,
            })

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        cached = self.data_cache[idx]
        sample = {'image': cached['image'], 'label': cached['label']}

        if self.transform and self.split == "train":
            sample = self.transform(sample)

        sample['case_name'] = cached['case_name']
        return sample
