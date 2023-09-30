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

from utils.data import collate_pool
from utils.model import CrystalGraphConvNet

# from inference import *


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if cuda:
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

        if i % print_freq == 0:
            if model_args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       accu=accuracies, prec=precisions, recall=recalls,
                       f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, pred in zip(test_cif_ids, test_preds):
                writer.writerow((cif_id, pred))
    else:
        star_label = '*'

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


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


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


##### START #####
dataset='matproj'
data_options=['/oasis/scratch/comet/txie/temp_project/data/matproj/20170419_semi.json', 'regression', 'band_gap']
atom_encoder='custom'
atom_encoder_param=['/home/txie/works/cgnn/Local/embeddings/elem_prop_embedding.json']
task='regression'
n_out=1
disable_cuda=False
workers=0
epochs=1000
start_epoch=0
batch_size=256
lr=0.05
lr_milestones=[800]
momentum=0.9
weight_decay=0.0
print_freq=10
resume=''
train_size=16458
val_size=5485
test_size=5486
optim='SGD'
atom_fea_len=64
h_fea_len=32
n_conv=4
n_h=1
cuda=False


if task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

modelpath = 'model/band-gap.pth.tar'
model_checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
model_args = argparse.Namespace(**model_checkpoint['args'])


from utils.data import CIFData_one
dataset = CIFData_one('data2/1000041.cif')

collate_fn = collate_pool
test_loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        num_workers=0, collate_fn=collate_fn,
                        pin_memory=False)
structures, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                            atom_fea_len=atom_fea_len,
                            n_conv=n_conv,
                            h_fea_len=h_fea_len,
                            n_h=n_h,
                            classification=True if task == 'classification' else False)

if cuda:
    model.cuda()

if task == 'classification':
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

val_loader = test_loader
test = False

result, calc_time = validate(test_loader, model, criterion, normalizer, test=True)

