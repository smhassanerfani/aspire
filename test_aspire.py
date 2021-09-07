
import os
import argparse
import numpy as np
from PIL import Image
from skimage.io import imsave

import torch
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from model.psp import PSPNet
# from model.sp import SPNet
# from model.ft import FTNet
# from model.fer import FERNet
# from model.ema import EMANet
# from model.ocr import OCRNet
# from model.oc import OCNet
# from model.deeplabv3 import DeepLabV3
# from model.cc import CCNet
# from model.acf import ACFNet
# from model.da import DANet
# from model.ann import ANNet

from AtlantisLoader import AspireDataSet



palette = [0,0,0,128,0,0,0,128,0,128,128,0,0,0,128,128,0,128,0,128,128,128,128,128,64,0,0,192,0,0,64,128,0,192,128,0,
           64,0,128,192,0,128,64,128,128,192,128,128,0,64,0,128,64,0,0,192,0,128,192,0,0,64,128,128,64,128,0,192,128,
           128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,128,192,192,128,0,0,64,128,0,
           64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,
           192,192,0,192,64,128,192,192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,
           128,192,192,64,64,64]

id_to_colorid = {3:  0,  4:  1,  7:  2,  9:  3, 10:  4, 11:  5, 12:  6, 13:  7, 16:  8, 17:  9,
                18: 10, 19: 11, 20: 12, 21: 13, 22: 14, 23: 15, 24: 16, 26: 17, 29: 18, 30: 19,
                32: 20, 33: 21, 34: 22, 35: 23, 36: 24, 38: 25, 39: 26, 40: 27, 42: 28, 43: 29,
                44: 30, 45: 31, 53: 32, 54: 33, 55: 34, 56: 35,  1: 36,  2: 37,  5: 38,  6: 39,
                 8: 40, 14: 41, 15: 42, 25: 43, 27: 44, 28: 45, 31: 46, 37: 47, 41: 48, 46: 49,
                47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55}

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask, num_classes):
    mask_copy = np.zeros_like(mask)
    if num_classes==56:
        for k, v in id_to_colorid.items():
            mask_copy[mask == (k-1)] = v
    else:
        mask_copy = mask
    new_mask = Image.fromarray(mask_copy.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

MODEL = 'PSPNet'
NAME = 'aspire_psp'
SPLIT = 'test'
NUM_CLASSES = 1
BATCH_SIZE = 1
NUM_WORKERS = 1
PADDING_SIZE = '960'
DATA_DIRECTORY = './data/june_22/'
SAVE_PATH = './result/'+str(NUM_CLASSES)+'/'+SPLIT+'/'+NAME+'v2_'
RESTORE_FROM = './snapshots/'+NAME+'/epoch10.pth'


def get_arguments():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--padding-size", type=int, default=PADDING_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : PSPNet")
    parser.add_argument("--split", type=str, default=SPLIT,
                        help="test or val")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

args = get_arguments()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, total_steps):
    lr = lr_poly(args.learning_rate, i_iter, total_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'SPNet':
        model = SPNet(num_classes=args.num_classes)
    if args.model == 'DANet':
        model = DANet(num_classes=args.num_classes)
    if args.model == 'ANNet':
        model = ANNet(num_classes=args.num_classes)
    if args.model == 'EMANet':
        model = EMANet(num_classes=args.num_classes)
    if args.model == 'FENet':
        model = FENet(num_classes=args.num_classes)
    if args.model == 'ACFNet':
        model = ACFNet(num_classes=args.num_classes)
    if args.model == 'FERNet':
        model = FERNet(num_classes=args.num_classes)
    if args.model == 'FTNet':
        model = FTNet(num_classes=args.num_classes)
    if args.model == 'CCNet':
        model = CCNet(num_classes=args.num_classes)
    if args.model == 'DeepLabV3':
        model = DeepLabV3(num_classes=args.num_classes)
    if args.model == 'OCRNet':
        model = OCRNet(num_classes=args.num_classes)
    if args.model == 'OCNet':
        model = OCNet(num_classes=args.num_classes)



    model.eval()
    model.cuda()

    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    cudnn.enabled = True
    cudnn.benchmark = True


    testloader = data.DataLoader(
            AspireDataSet(args.data_dir, split=args.split, padding_size = args.padding_size),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    interp = nn.Upsample(size=(args.padding_size, args.padding_size), mode='bilinear', align_corners=True)


    for images, labels, name, width, height in testloader:
    # for images, name, width, height in testloader:
        images = images.cuda()
        images = F.upsample(images, [640, 640], mode='bilinear')
        with torch.no_grad():
            _, pred = model(images)
        pred = interp(pred).squeeze(1).cpu().data[0].numpy()
        # pred_copy = p
        # pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
        pred[pred>0.5]=1
        pred[pred<=0.5]=0


        top_pad = args.padding_size - height
        right_pad = args.padding_size - width
        pred = pred[top_pad:, :-right_pad]

        imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred)


if __name__ == '__main__':
    main()
    # os.system('python compute_iou.py --split ' + SPLIT + ' --pred_dir ' + SAVE_PATH + ' --num-classes ' + str(NUM_CLASSES))
