import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # learning arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--nusers', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate client')
    parser.add_argument('--lr_server', type=float, default=0.001, help='learning rate of server')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--agg', type=str, default='avg', help='averaging strategy')
    parser.add_argument('--epsilon', type=float, default=1.2, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')
    parser.add_argument('--gamma', type=float, default=0.7, help='iter parameter (default: 0.7)')
    # model arguments

    # other arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name')
    parser.add_argument('--iid', type=int, default=1, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--resume_snapshot', type=str, default='')

    parser.add_argument('--opt', type=str, default='normal')
    args = parser.parse_args()
    return args
