import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from itertools import cycle
import clip
import pdb
from PMP.pmp_cali import CustomCLIP
from PIL import Image
import converter_dassl, converter_domainbed
from utils import accuracy, AverageMeter, ProgressMeter, TensorboardWriter, ForeverDataIterator

class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.model = model
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self):
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)

    def train(self, mode=True):
        self.moving_avg.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    @property
    def module(self):
        return self.moving_avg.module


def get_dataset(args):
    if args.task == "domain_shift":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2)
        train_class_names = class_names
        train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        ]
        template = "a photo of a {}."
    
    elif args.task == "open_class":
        # load dassl data
        train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template = \
            converter_dassl.get_dassl_datasets(dataset_name=args.data, root=args.root, n_shot=args.n_shot)
        train_class_names = base_class_names
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        train_iter = ForeverDataIterator(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(open_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": open_class_names
            }
        ]

    elif args.task == "in_the_wild":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2, seed=args.seed, open_ratio=0.5)
        train_class_names = base_class_names
        train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."
    
    return train_iter, val_loader, test_loaders, train_class_names, template

def get_dataset_source(args):
    if args.task == "domain_shift":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.source, root=args.root, targets=args.source_targets, holdout=0.2)
        train_class_names = class_names
        train_iter = {
                "iter":converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers),
                "class_names": class_names
            }
        val_loader = {
                "loader":DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        ]
        template = "a photo of a {}."
    
    elif args.task == "open_class":
        # load dassl data
        train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template = \
            converter_dassl.get_dassl_datasets(dataset_name=args.source, root=args.root, n_shot=args.n_shot)
        train_class_names = base_class_names
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        train_iter = {
                "iter":ForeverDataIterator(train_loader),
                "class_names": class_names
            }
        val_loader ={
                "loader":DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(open_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": open_class_names
            }
        ]

    elif args.task == "in_the_wild":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.source, root=args.root, targets=args.source_targets, holdout=0.2, seed=args.seed, open_ratio=0.5)
        train_class_names = base_class_names
        train_iter = {
                "iter":converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers),
                "class_names": class_names
            }
        val_loader ={
                "loader":DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."
    
    return train_iter, val_loader, test_loaders, train_class_names, template

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def ptm_train(train_iter: dict,source_train_iter: dict, model:CustomCLIP, moving_avg_model: GeneralMovingAverage,
          optimizer, lr_scheduler, epoch: int, args, writer: TensorboardWriter, device,threshold=0.922):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    pseu_accs = AverageMeter('Pseu Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs,pseu_accs],
        prefix="Epoch: [{}]".format(epoch))

    # Freeze all Norm Layers
    model.eval()
    
    prefixes,suffixes=None,None
    if(type(train_iter)==dict):
        if "prefixes" in train_iter:
            prefixes,suffixes=train_iter["prefixes"],train_iter["suffixes"]
        train_iter=train_iter["iter"]
        
    s_prefixes,s_suffixes=None,None
    if(type(train_iter)==dict):
        if "prefixes" in source_train_iter:
            s_prefixes,s_suffixes=source_train_iter["prefixes"],source_train_iter["suffixes"]
        source_train_iter=source_train_iter["iter"]

    end = time.time()
    for i in range(args.iters_per_epoch):
        # obtain data
        if args.task in ["domain_shift", "in_the_wild"]:
            x, labels = [], []
            for x_d, labels_d in next(train_iter):
                x.append(x_d)
                labels.append(labels_d)
            x, labels = torch.cat(x), torch.cat(labels)
            s_x, s_labels = [], []
            for x_d, labels_d in next(source_train_iter):
                s_x.append(x_d)
                s_labels.append(labels_d)
            s_x, s_labels = torch.cat(x), torch.cat(labels)
        else:
            x, labels = next(train_iter)
            s_x, s_labels=next(train_iter)
        x, labels = x.to(device), labels.to(device)
        # measure data loading time
        data_time_step = time.time() - end
        data_time.update(data_time_step)

        # compute output
        y,loss_cali=model(x,prefixes,suffixes)
        y_softmax = F.softmax(y, dim=1)

        # Copy y_softmax to pseudo_labels
        pseudo_labels = torch.argmax(y_softmax, dim=1)

        # Identify confident predictions
        confident_mask = torch.max(y_softmax, dim=1)[0] > threshold
        loss_ = F.cross_entropy(y, pseudo_labels, reduction='none')
        loss = torch.masked_select(loss_, confident_mask)
        
        cls_acc = accuracy(y, labels)[0]
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        if(loss.shape[0]!=0):
            loss=loss.sum()#+loss_cali
            losses.update(loss.item(), x.size(0))
            # loss.backward()
            # model.prompt_learner.Step()
            tmp=pseudo_labels[confident_mask]==labels[confident_mask]
            pseu_accs.update(tmp.sum()/len(tmp),len(tmp))
            
            x, labels = s_x.to(device), s_labels.to(device)
            y,loss_cali=model(x,s_prefixes,s_suffixes,source=True)
            loss=F.cross_entropy(y, labels)#+loss_cali
            loss.backward()
            model.prompt_learner.Step()


        moving_avg_model.update()

        bma_weight = moving_avg_model.weight

        # measure elapsed time
        batch_time_step = time.time() - end
        batch_time.update(batch_time_step)

        writer.record_training_values(
            {
                "Acc@1": (cls_acc.item(), x.shape[0]),
                "Time": (batch_time_step,),
                "Data": (data_time_step,),
                "Weight": (bma_weight,),
            }
        )

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate_ptm(val_loader,source_train_iter, model:CustomCLIP, args, device, shift=0,threshold=0.92,train=False) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pseu_accs = AverageMeter('pseu_accs', ':6.2f')
    if train:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1,pseu_accs],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1],
            prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    # pdb.set_trace()
    prefixes,suffixes=None,None
    if type(val_loader)==dict:
        if "prefixes" in val_loader:
            prefixes,suffixes=val_loader["prefixes"],val_loader["suffixes"]
        val_loader=val_loader["loader"]
        
    s_prefixes,s_suffixes=None,None
    if type(source_train_iter)==dict:
        if "prefixes" in source_train_iter:
            s_prefixes,s_suffixes=source_train_iter["prefixes"],source_train_iter["suffixes"]
        source_train_iter=source_train_iter["iter"]
    
    end = time.time()
    # pdb.set_trace()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device) - shift

        if train:
            # pdb.set_trace()
            output_similarity,loss_cali = model(images,prefixes,suffixes)
            acc1, = accuracy(output_similarity, target, topk=(1,))
            
            y_softmax = F.softmax(output_similarity, dim=1)
            pseudo_labels = torch.argmax(output_similarity, dim=1)
            # Identify confident predictions
            confident_mask = torch.max(y_softmax, dim=1)[0] > threshold
            loss_ = F.cross_entropy(output_similarity, pseudo_labels, reduction='none')
            # loss_ = F.cross_entropy(output_similarity, target, reduction='none')
            loss = torch.masked_select(loss_, confident_mask)
            
            if(loss.shape[0]!=0):
                loss=loss.sum()
                loss.backward()
                model.prompt_learner.Step()
                tmp=pseudo_labels[confident_mask]==target[confident_mask]
                pseu_accs.update(tmp.sum()/len(tmp),len(tmp))
                
                s_x,s_labels=[],[]
                for x_d, labels_d in next(source_train_iter):
                    s_x.append(x_d)
                    s_labels.append(labels_d)
                s_images, s_target = torch.cat(s_x), torch.cat(s_labels)
                images = s_images.to(device)
                target = s_target.to(device) - shift
                output_similarity,loss_cali = model(images,s_prefixes,s_suffixes,source=True)
                loss = F.cross_entropy(output_similarity, target)+loss_cali
                loss.backward()
                model.prompt_learner.Step()
        else:
            with torch.no_grad():
                output_similarity,_ = model(images,prefixes,suffixes)
                acc1, = accuracy(output_similarity, target, topk=(1,))
                
        top1.update(acc1.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        # if i==1:
        #     tmp=images[0].detach().cpu().numpy().transpose(1,2,0)
        #     tmp=(tmp*255).astype(np.uint8)
        #     img=Image.fromarray(tmp)
        #     img.save("test2.jpg")
        #     pdb.set_trace()

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return top1.avg

def evaluate_all_ptm(model:CustomCLIP, val_loader,source_train_iter, test_loaders,source_test_loaders, args, writer, device,threshold=0.922,train=False):
    print("Evaluate on validation set...")
    val_acc1 = validate_ptm(val_loader,source_train_iter, model, args, device,threshold=threshold,train=train)
    # writer.write_eval_values({"Acc@1": val_acc1}, prefix="val")
    # pdb.set_trace()
    for test_loader,source_test_loader in zip(test_loaders,cycle(source_test_loaders)):
        split_name = test_loader["name"]
        print(f"Evaluate on {split_name} set...")
        validate_ptm(test_loader,source_train_iter,model, args, device,threshold=threshold,train=train)
        writer.write_eval_values({"Acc@1": val_acc1}, prefix=split_name)

    return val_acc1