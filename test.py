import argparse
import os
import torch
from torch.autograd import Variable
import numpy as np
from dataset import PointcloudPatchDataset, SequentialPointcloudPatchSampler
import shutil
def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='./data/pointCleanNetDataset',
                        help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./results',
                        help='output folder (point clouds)')
    parser.add_argument('--test_set',type=str,default='validationset.txt',help='test set file name')
    parser.add_argument('--patch_radius', type=float, default=0.05, nargs='+')
    parser.add_argument('--batchsize', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--iter', type=int,
                        default=2, help='iterations')
    parser.add_argument('--points_per_patch', type=int,
                        default=500)
    parser.add_argument('--workers', type=int, default=6,
                        help='number of data loading workers - 0 means same thread as main execution')
    return parser.parse_args()

def get_data(opt,test_file):

    dataset=PointcloudPatchDataset(
        root=opt.outdir,
        shape_names=test_file,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
    )

    datasampler = SequentialPointcloudPatchSampler(
        dataset)

    dataloader=torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=opt.batchsize,
        num_workers=int(opt.workers)
    )
    return dataloader,datasampler,dataset


def test_penet(opt):

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    metapcd = torch.load('./model/model.pth')
    metapcd.cuda()
    metapcd.eval()
    with open(os.path.join(opt.indir,opt.test_set)) as f:
        shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))
    for iter in range(opt.iter):
        if iter==0:
            for shape in shape_names:
                source_file=os.path.join(opt.indir,shape+'.xyz')
                target_file=os.path.join(opt.outdir,shape+'_0.xyz')
                shutil.copy(source_file,target_file)
                source_clean=os.path.join(opt.indir,shape+'.clean_xyz')
                target_clean = os.path.join(opt.outdir, shape + '_0.clean_xyz')
                shutil.copy(source_clean,target_clean)
        for shape in shape_names:
            save_file=opt.outdir+'/'+shape + '_'+str(iter + 1) + '.xyz'
            if not os.path.exists(save_file):
                shape = shape + '_' + str(iter)
                test_dataloader, test_datasampler, test_dataset = get_data(opt, [shape])
                test_batches = enumerate(test_dataloader, 0)
                print('load done')
                total_test_batches = len(test_dataloader)
                out_put = []
                import time
                t1 = time.time()
                for batch_index, data in test_batches:
                    if batch_index % 200 == 0:
                        print('%s %d/%d time: %f' % (shape.strip(), batch_index, total_test_batches, time.time() - t1))
                        t1 = time.time()
                    points = data[0]
                    points = Variable(points)
                    points = points.cuda()
                    patch_radius = data[2][0]
                    patch_radius = patch_radius.cuda()
                    xyz = data[2][1]
                    xyz = xyz.cuda()
                    noise_set = metapcd(points)
                    noise_set = noise_set.transpose(1, 2)
                    for b, noise in enumerate(noise_set):
                        out = xyz[b:b + 1, :] + noise * patch_radius[b]
                        out_put.append((out).data.cpu().numpy())
                    # patch=points.data.cpu().numpy()*patch_radius.cpu().numpy()+2.5
                    # np.savetxt(os.path.join(opt.outdir, 'patch.txt'), patch[0,...])
                out_put = np.concatenate(out_put, 0)
                np.savetxt(save_file, out_put)
                shutil.copy(os.path.join(opt.outdir, shape + '.clean_xyz'),
                            os.path.join(opt.outdir, shape[:-1] + str(iter + 1) + '.clean_xyz'))





def main():
    train_opt=parse_arguments()
    test_penet(train_opt)

    print(1)
if __name__ == '__main__':
    main()
