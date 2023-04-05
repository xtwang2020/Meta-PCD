# coding:utf-8
import os
import csv
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm

def distance_min(center, points, k=1, squera=True):
    import scipy.spatial as spatial
    kdtree = spatial.cKDTree(points, 10)
    D_min, _ = kdtree.query(center, k)
    if squera:
        return D_min * D_min
    else:
        return D_min


def Chamfer_distance(gt, prediction):
    pre_gt = np.mean(distance_min(prediction, gt, squera=True))
    gt_pre = np.mean(distance_min(gt, prediction, squera=True))
    return pre_gt + gt_pre


def Hausdorff_distance(gt, prediction):
    pre_gt = np.max(distance_min(prediction, gt, squera=False))
    gt_pre = np.max(distance_min(gt, prediction, squera=False))
    return np.max([pre_gt, gt_pre])


def RMSD(gt, prediction):  # root mean square distance to surface (RMSD) of each point
    return np.sqrt(np.mean(distance_min(prediction, gt, squera=True)))


def Normal_CD(gt, prediction, normal):
    kdtree = spatial.cKDTree(gt, 10)
    D_min, idx = kdtree.query(prediction, 1)
    p = gt[idx]
    n = normal[idx, :3]
    n_dis = np.abs(np.sum((prediction - p) * n, -1))
    n_d = np.mean(n_dis)
    return n_d

def hard_easy(file):
    hard_list=['galera','dragon','happy','column_head','netsuke']
    easy_list = ['cylinder','boxunion2','box_push','star_smooth','icosahedron']
    for shape in hard_list:
        if shape in file:
            return True
    return False
def denoise_metric():
    label_dir = './data/pointCleanNetDataset/'
    eval_dir = {

        'MetaPCD': './results',
    }
    result_dir = './'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_dir + '/metric.csv', 'w', encoding='utf-8',newline='') as f:
        csv_writer = csv.writer(f)
        head = ['Method','iteration', 'CD', 'noise=0', 'noise=0.005', 'noise=0.01', 'noise=0.025', 'easy', 'hard']
        csv_writer.writerow(head)
        for method in eval_dir:
            for iter in range(1, 3):
                print('method: %s iter: %d'% (method, iter))
                file_list = os.listdir(eval_dir[method])
                Results_cd = np.zeros([7])
                Results_num = np.zeros([7])
                for file in tqdm(file_list):
                    if file[-4:] == '.xyz' and int(file[-5])==iter:
                        # print(file)
                        gt = np.loadtxt(label_dir + '/' + file[:-6]+'.clean_xyz')
                        ddiag=np.linalg.norm(gt.max(0)-gt.min(0))
                        center=(gt.max(0)+gt.min(0))/2
                        gt=(gt-center)/ddiag
                        # print(file)
                        # prediction=np.loadtxt(eval_dir[method] + '/' + file)

                        prediction=np.loadtxt(eval_dir[method] + '/' + file)[:,:3]
                        prediction=(prediction-center)/ddiag
                        cd = Chamfer_distance(gt, prediction)
                        Results_cd[0] = Results_cd[0] + cd
                        Results_num[0] = Results_num[ 0] + 1
                        if '2.50e-02' in file:
                            Results_cd[4] = Results_cd[4] + cd
                            Results_num[4] = Results_num[4] + 1
                        elif '1.00e-02' in file:
                            Results_cd[3] = Results_cd[ 3] + cd
                            Results_num[3] = Results_num[ 3] + 1
                        elif '5.00e-03' in file:
                            Results_cd[ 2] = Results_cd[2] + cd
                            Results_num[ 2] = Results_num[2] + 1
                        else:
                            Results_cd[1] = Results_cd[1] + cd
                            Results_num[1] = Results_num[1] + 1
                        if hard_easy(file):
                            Results_cd[6] = Results_cd[6] + cd
                            Results_num[6] = Results_num[ 6] + 1
                        else:
                            Results_cd[5] = Results_cd[5] + cd
                            Results_num[5] = Results_num[ 5] + 1
                Results=Results_cd/Results_num
                line = [method,iter]
                line = line + [r for r in Results]
                csv_writer.writerow(line)
                print(line)

def main():
    denoise_metric()


if __name__ == '__main__':
    main()
