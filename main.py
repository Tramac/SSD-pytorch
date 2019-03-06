import argparse
import torch

import torch.utils.data as data

from dataset.config import VOC_ROOT, MEANS, voc
from dataset.voc import VOCDetection, VOCAnnotationTransform, BaseTransform, detection_collate
from utils.augmentations import SSDAugmentation
from ssd.ssd300 import build_ssd
from ssd.utils_ssd.multiboxloss import MultiBoxLoss
from trainer import Trainer
from evaluator import Evaluator


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = voc
    train_dataset = VOCDetection(root=config.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    test_dataset = VOCDetection(config.dataset_root, [('2007', 'test')], BaseTransform(300, MEANS),
                                VOCAnnotationTransform())

    train_data_loader = data.DataLoader(train_dataset, config.batch_size, num_workers=config.num_workers, shuffle=True,
                                        collate_fn=detection_collate, pin_memory=True)

    model = build_ssd(config.mode, cfg['min_dim'], cfg['num_classes']).to(device)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, config.cuda)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    trainer = Trainer(model, criterion, optimizer, train_data_loader, config, device)

    evaluator = Evaluator(model, test_dataset, config, device)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        evaluator.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--dataset', default='VOC2007', choices=['VOC2007', 'VOC2012', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=VOC_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=str2bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--mode', default='train', type=str,
                        help='train/test')

    # For eval
    parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open')

    args = parser.parse_args()

    main(args)
