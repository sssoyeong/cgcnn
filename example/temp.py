import argparse
import os
import sys
import torch

def main():
    # parser
    parser = argparse.ArgumentParser(description='Crystal Gated Neural Networks (CGCNN) - inference')
    parser.add_argument('model', help='select pre-trained model. \
        formation-energy-per-atom, final-energy-per-atom,band-gap, efermi, bulk-moduli, shear-moduli, poisson-ratio')
    parser.add_argument('cifinput', help='input file (.cif)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args(sys.argv[1:])

    # print(type(args))
    # print(dir(args))
    # args_dict = vars(args)
    # print(args_dict)
    # print(type(args))

    modelpath = 'model/'+args.model+'.pth.tar'
    model_checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print(model_args)


if __name__ == '__main__':
    main()