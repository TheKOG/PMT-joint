import random
import warnings
import argparse
import shutil
import scipy as sp

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import clip

from utils import CompleteLogger, TensorboardWriter
from engine_cali import GeneralMovingAverage, get_dataset, get_dataset_source, ptm_train, evaluate_all_ptm
from PMP.pmp_cali import CustomCLIP
import PMP.pmp
from engine import evaluate_all_pmp

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import yacs.config as CN
import yaml

def setup_cfg(args):
    # 读取YAML文件
    with open(args.pmp_cfg_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # 将字典转换为CfgNode
    cfg = CN.CfgNode(config_dict, new_allowed=True)

    return cfg
import os
import pdb
def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    # print(args)
    threshold=args.threshold
    debug=args.debug
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    
    best_pth=logger.get_checkpoint_path('best')
    clip_model, _ = clip.load(args.arch, device)
    clip_model.requires_grad_(False)
    
    train_iter, val_loader, test_loaders, train_class_names, template = get_dataset(args)
    
    source_train_iter, source_val_loader, source_test_loaders, source_train_class_names, source_template = get_dataset_source(args)
    
    clip_ptm=CustomCLIP(clip_model=clip_model,classnames=train_class_names,source_classnames=source_train_class_names,cfg=setup_cfg(args),device=device)
    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    classifier = clip_model.visual
    
    if(os.path.isfile(best_pth)):
        classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
        
    # clip.model.convert_weights(classifier)
    # pdb.set_trace()
    classifier.eval()
    
    # obtain text features
    
    for test_loader in test_loaders:
        test_loader["prefixes"],test_loader["suffixes"]=clip_ptm.prompt_learner.get_pre_suf(test_loader["class_names"],clip_model)
        # test_loader["text_features"] = get_text_features(clip_model, template, test_loader["class_names"], device)
        
    for s_test_loader in source_test_loaders:
        s_test_loader["prefixes"],s_test_loader["suffixes"]=clip_ptm.prompt_learner.get_pre_suf(s_test_loader["class_names"],clip_model)
    # pdb.set_trace()
    source_val_loader["prefixes"],source_val_loader["suffixes"]=clip_ptm.prompt_learner.get_pre_suf(source_train_class_names,clip_model)
    source_train_iter["prefixes"],source_train_iter["suffixes"]=source_val_loader["prefixes"],source_val_loader["suffixes"]

    # define beta moving average
    beta_dist = sp.stats.beta(args.beta, args.beta)
    total_iter = args.epochs * args.iters_per_epoch
    weight_func = lambda it: beta_dist.pdf((it + 0.5) / (total_iter + 1))

    bma_classifier = GeneralMovingAverage(classifier, weight_func)

    if args.phase == "train":
        # define optimizer and lr scheduler
        optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)
        
        # define temperature for training
        if args.temperature is None:
            args.temperature = clip_model.logit_scale.exp().item()

        # define tensorboard writer
        writer = TensorboardWriter(args.log, flush_freq=20)

        # evaluate zero-shot performance
        best_val_acc1 = evaluate_all_ptm(clip_ptm, val_loader,source_train_iter, [],source_test_loaders, args, writer, device,threshold=threshold,train=False)

        # start training
        for epoch in range(args.epochs):
            # evaluate all
            val_acc1 = evaluate_all_ptm(clip_ptm, val_loader,source_train_iter, [],source_test_loaders, args, writer, device,threshold=threshold,train=True)
            # continue
            if val_acc1 > best_val_acc1:
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
                best_val_acc1 = val_acc1
                clip_ptm.prompt_learner.Save()

        print("Training completed.")
        
    # clip_ptm=CustomCLIP(clip_model=clip_model,classnames=train_class_names,source_classnames=source_train_class_names,cfg=setup_cfg(args),device=device)
    # print("Evaluate best model:")
    # evaluate_all_ptm(clip_ptm, val_loader,source_train_iter, test_loaders,source_test_loaders, args, writer, device,threshold=threshold,train=False)
    
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-s', '--source', metavar='SOURCE', default='PACS')
    parser.add_argument('--source-targets', nargs='+', type=int, default=[3],
                        help='target domain(s) (DomainBed datasets only)')
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet')
    parser.add_argument('--task', default='domain_shift', choices=
                        ['domain_shift', 'open_class', 'in_the_wild'])
    parser.add_argument('--targets', nargs='+', type=int, default=None,
                        help='target domain(s) (DomainBed datasets only)')
    parser.add_argument('--n-shot', type=int, default=0)
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=5e-6, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        metavar='W', help='weight decay (default: 0.1)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--log', type=str, default='exp0',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    # parameters for CLIPood
    parser.add_argument('--temperature', type=float, default=None, help=
                        "Use CLIP's original temperature in default.")
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.5)
    
    parser.add_argument('--pmp_cfg_file', type=str, default='./configs/vit_b16_c16_ptm.yaml', help='Path to the pmp_cfg_file YAML file.')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold of pseudo labels.')
    parser.add_argument('--debug', type=bool, default=True, help='Debug')

    args = parser.parse_args()
    main(args)
    print("done")
