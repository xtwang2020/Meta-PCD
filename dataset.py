import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import time

def load_shape(point_filename,clean_filename):
    pts = np.loadtxt(point_filename).astype(np.float32)
    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10))))
    # t1 = time.time()
    kdtree=spatial.cKDTree(pts)
    # print(time.time() - t1)
    clean_points=np.loadtxt(clean_filename).astype(np.float32)
    clean_kdtree=spatial.cKDTree(clean_points)


    return Shape(pts=pts,kdtree=kdtree,
                 clean_points=clean_points,
                 clean_kdtree=clean_kdtree)

class Shape():
    def __init__(self,pts,kdtree,clean_points=None, clean_kdtree=None):
        self.pts = pts
        self.clean_points = clean_points
        self.kdtree = kdtree
        self.clean_kdtree = clean_kdtree

class PointcloudPatchDataset(data.Dataset):
    def __init__(self, root, patch_radius, points_per_patch, shape_list_file=None, shape_names=None):
        self.root=root
        self.shape_list_file=shape_list_file
        self.patch_radius=np.array(patch_radius)
        self.points_per_patch=points_per_patch
        self.shapes=[]
        self.shape_idx=-1
        # self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)
        # get all shape names in the dataset
        if shape_names is not None:
            self.shape_names = shape_names
        else:
            self.shape_names = []
            with open(os.path.join(root, self.shape_list_file)) as f:
                self.shape_names = f.readlines()
            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))
        self.rng = np.random.RandomState()

        # get basic information for each shape in the dataset
        self.shape_patch_count=[]
        self.patch_radius_absolute=[]
        for shape_idx, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))
            # load from text file and save in more efficient numpy format
            # shape=self.shape_cache.get(shape_idx)
            self.shapes.append(self.load_shape_by_index(shape_idx))
            self.shape_patch_count.append(self.shapes[-1].pts.shape[0])
            bbdiag = float(np.linalg.norm(self.shapes[-1].pts.max(0) - self.shapes[-1].pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag*self.patch_radius)

    def select_patch_points(self, patch_pts, patch_radius, center_point_idx, shape, clean_points=False):
        #select patch points
        if clean_points:
            #patch_point_inds = np.array(shape.clean_kdtree.query_ball_point(shape.clean_points[center_point_ind, :], patch_radius))
            dis,patch_point_idx = shape.clean_kdtree.query(shape.pts[center_point_idx, :], 500)
        else:
            patch_point_idx = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_idx, :], patch_radius))
        if len(patch_point_idx) > 0:
            point_count = min(self.points_per_patch, len(patch_point_idx))
            if point_count < len(patch_point_idx):
                iii=self.rng.choice(len(patch_point_idx), point_count, replace=False)
                iii.sort()
                patch_point_idx = patch_point_idx[iii]
                patch_point_idx[0]=center_point_idx
            start = 0
            end = start + point_count
            if clean_points:
                points_base = shape.clean_points
            else:
                points_base = shape.pts
            patch_pts[start:end, :] = torch.from_numpy(points_base[patch_point_idx, :])
            patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_idx, :])
            patch_pts=patch_pts/patch_radius
        return patch_pts

    def __getitem__(self, index):
        shape_idx, center_point_idx=self.shape_index(index)
        shape = self.shapes[shape_idx]
        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.FloatTensor(self.points_per_patch, 3).zero_()
        patch_pts = self.select_patch_points(patch_pts,
                                             self.patch_radius_absolute[shape_idx],
                                             center_point_idx,
                                             shape)
        clean_pts = torch.FloatTensor(self.points_per_patch, 3).zero_()
        clean_pts = self.select_patch_points(clean_pts,self.patch_radius_absolute[shape_idx],center_point_idx,shape,clean_points=True)
        center_point = torch.from_numpy(shape.pts[center_point_idx, :])
        radius = torch.FloatTensor([self.patch_radius_absolute[shape_idx]])
        return patch_pts, clean_pts, (radius, center_point )

        #patch_pts:[x,y,z,m_x,m_y,m_z]


    def __len__(self):
        return sum(self.shape_patch_count)

    def shape_index(self,index):
        shape_patch_offset = 0
        shape_idx=None
        for shape_idx, shape_patch_count in enumerate(self.shape_patch_count):
            if index>=shape_patch_offset and index<shape_patch_offset+shape_patch_count:
                shape_patch_idx=index-shape_patch_offset
                break
            shape_patch_offset=shape_patch_offset+shape_patch_count
        return shape_idx, shape_patch_idx

    def load_shape_by_index(self,shape_idx):
        point_filename=os.path.join(self.root,self.shape_names[shape_idx]+'.xyz')
        clean_filename=os.path.join(self.root,self.shape_names[shape_idx]+'.clean_xyz')
        return load_shape(point_filename, clean_filename,)



class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        # if self.seed is None:
        #     self.seed = np.random.random_integers(0, 2**32-1, 1)
        self.rng = np.random.RandomState()

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count