import argparse
import os
import shutil
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.data import CIFData_one
from utils.data import collate_pool
from utils.model import CrystalGraphConvNet


def main():
    global args, model_args, best_mae_error

    # parser
    parser = argparse.ArgumentParser(description='Crystal Gated Neural Networks (CGCNN) - inference')
    parser.add_argument('model', help='select pre-trained model. \
                                    formation-energy-per-atom, final-energy-per-atom,band-gap, efermi, \
                                    bulk-moduli, shear-moduli, poisson-ratio')
    parser.add_argument('cifpath', help='path to the CIF files. ')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    args = parser.parse_args(sys.argv[1:])

    # model file check
    modelpath = 'model/'+args.model+'.pth.tar'
    if os.path.isfile(modelpath):
        print("=> loading model params '{}'".format(args.model))
        model_checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        
        print("=> loaded model params '{}'".format(args.model))
    else:
        print("=> no model found like '{}'".format(args.model))
    
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    if model_args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.

    # load data
    dataset = CIFData_one(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # build model
    structures, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task == 'classification' else False)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(modelpath):
        print("=> loading model '{}'".format(modelpath))
        checkpoint = torch.load(modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(modelpath))

    # prediction
    result, calc_time = validate(test_loader, model, criterion, normalizer, test=True)
    print(f'=> prediction complete (elapsed time {calc_time:.5f} sec)')
    print(f'   {result}')

    # save .json
    file_path = f"./result_{args.model}_{time.strftime('%y%m%d%H%M%S')}.json"

    result_json = {}
    result_json['result'], result_json['calculation_time [sec]'] = [], calc_time
    for idx, key in enumerate(result):
        result_json['result'].append({
            f'{key}.cif': result[key]
        })
    with open(file_path, 'w') as outfile:
        json.dump(result_json, outfile, indent=4)

    if os.path.exists(file_path):
        print(f'=> result file \'{file_path}\' saved successfully')


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

        # compute output
        output = model(*input_var)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            if test:
                test_pred = torch.exp(output.data.cpu())
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        result_dict = dict.fromkeys(test_cif_ids)
        for idx, key in enumerate(result_dict):
            result_dict[key] = test_preds[idx]
    return result_dict, batch_time.val


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
