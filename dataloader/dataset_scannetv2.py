# -*- coding: utf-8 -*-
"""
@File: dataset_scannetv2.py
@Author:Huitong Jin
@Date:2023/2/9
"""

# =============================
# imports and global variables
# =============================

import os
import numpy as np
import torch
from torch.utils import data
import numba as nb


# ===============
# utils functions
# ===============

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


# ================
# Define PLY types
# ================
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

ScannetV2_label_name = {0: 'unclassified',
                        1: 'wall',
                        2: 'floor',
                        3: 'cabinet',
                        4: 'bed',
                        5: 'chair',
                        6: 'sofa',
                        7: 'table',
                        8: 'door',
                        9: 'window',
                        10: 'bookshelf',
                        11: 'picture',
                        12: 'counter',
                        14: 'desk',
                        16: 'curtain',
                        24: 'refridgerator',
                        28: 'shower curtain',
                        33: 'toilet',
                        34: 'sink',
                        36: 'bathtub',
                        39: 'otherfurniture'}

# ===================
# Numpy reader format
# ===================
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


# =========
# Functions
# =========
def parse_header(plyfile, ext):
    # Variable
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])
        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variable
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)
    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    points = np.random.rand(5, 3)
    values = np.random.randint(2, size=10)
    write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    data = read_ply('example.ply')
    values = data['values']
    array([0, 0, 1, 1, 0])

    points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:
        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')
        # Get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')
        # Get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:
            # parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)
            # get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)
            # get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)
            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:
            # Parse header
            num_points, properties = parse_header(plyfile, ext)
            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, filed_names):
    # List of lines to write
    lines = []
    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])
    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, filed_names[i]))
            i += 1
    return lines


def collate_fn_points(data):
    print(data[0][0].shape)
    points_stack = np.concatenate([d[0] for d in data], axis=0).astype(np.float32)
    cloud_labels_all = np.concatenate([d[1] for d in data], axis=0)
    weakly_labels_stack = np.concatenate([d[2] for d in data], axis=0)
    gt_labels_stack = np.concatenate([d[3] for d in data], axis=0)
    mask = np.concatenate([d[4] for d in data], axis=0)
    return torch.from_numpy(points_stack), torch.from_numpy(cloud_labels_all), torch.from_numpy(
        weakly_labels_stack), torch.from_numpy(gt_labels_stack), torch.from_numpy(mask)


# ============================
# Define scannet dataset class
# ============================
class ScannetDataset(data.Dataset):
    def __init__(self, path, split='train', return_ref=False):
        super().__init__()
        self.return_ref = return_ref
        self.label_to_names = ScannetV2_label_name
        self.label_values = np.sort(np.sort([k for k, v in self.label_to_names.items()]))
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.data_root = path
        self.split = split
        self.block_size = 1.0
        self.ignored_label = np.sort([0])
        self.num_pos = 0
        self.num_neg = 0

        if split == 'train':
            self.clouds_path = np.loadtxt(os.path.join(self.data_root, 'scannetv2_train.txt'), dtype=np.str)
        else:
            self.clouds_path = np.loadtxt(os.path.join(self.data_root, 'scannetv2_val.txt'), dtype=np.str)
        self.train_path = '/data/dataset/scannet/input_0.040'
        self.files = np.sort([os.path.join(self.train_path, f + '.ply') for f in self.clouds_path])

    def __getitem__(self, index):
        path = self.files[index]
        # print(path)
        data = read_ply(path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        gt_labels = data['class'].astype(np.int32)

        data_tuple = (points, gt_labels)
        if self.return_ref:
            data_tuple += (points,)
        return data_tuple

    def __len__(self):
        return len(self.files)


class spherical_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 1.5], min_volume_space=[3, -np.pi, -3]):
        """Initialization"""
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        """Generates one sample of data"""
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        # data_tuple = (voxel_position,processed_label)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label, dtype=bool)
        valid_label[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label, axis=0)
        max_distance = max_bound[0] - intervals[0] * (max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2) - np.transpose(voxel_position[0], (1, 2, 0))
        distance_feature = np.transpose(distance_feature, (1, 2, 0))
        # convert to boolean feature
        distance_feature = (distance_feature > 0) * -1.
        distance_feature[grid_ind[:, 2], grid_ind[:, 0], grid_ind[:, 1]] = 1.

        data_tuple = (distance_feature, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


if __name__ == '__main__':
    data = ScannetDataset('/data/dataset/scannet', npoints=40000, split='val')
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=20, shuffle=False)
    Max = -1
    Min = 200000
    total_files_name = []
    masks = []
    labels_total = []
    val_prob = np.zeros(20, dtype=np.float32)
    for points, cloud_labels_all, weakly_label, gt_label, mask, files_name in train_dataloader:
        total_files_name.append(files_name)
        masks.append(mask)
        labels_total += [gt_label]
        num_points = points.shape[0]

        print(points.shape, weakly_label.shape, gt_label.shape)
    num_pos = data.num_pos
    num_neg = data.num_neg
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)
    print(num_pos / (num_pos + num_neg))
