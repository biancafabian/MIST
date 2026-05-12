import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import numpy as np

from utils.dataset_ACDC import ACDCdataset
from test_ACDC import inference
from lib.networks import MIST_CAM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to best.pth checkpoint")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--z_spacing", type=int, default=10)
parser.add_argument("--test_save_dir", default="./predictions")
parser.add_argument("--seed", type=int, default=2222)
args = parser.parse_args()

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

log_folder = os.path.join(os.path.dirname(args.checkpoint), "inference_log")
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, "inference.txt"), level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size, args.img_size),
               img_size_s2=(224, 224), model_scale='small',
               decoder_aggregation='additive', interpolation='bilinear').cuda()
net.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False))
net.eval()
print("Checkpoint loaded:", args.checkpoint)

test_save_path = os.path.join(os.path.dirname(args.checkpoint), args.test_save_dir)
os.makedirs(test_save_path, exist_ok=True)

db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

results = inference(args, net, testloader, test_save_path)
print("mean_dice: %f  mean_hd95: %f  mean_jacard: %f  mean_asd: %f" % results)
