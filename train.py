import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import pdb
import torch.nn as nn
import random
import torch
import torch.nn.functional as F
from textmel_dataset import TextMelLoader, TextAudioCollate
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import hparams as hp
import pdb
############################


def main():
    print('Initializing Training Process..')
    train_dataset = TextMelLoader(hp)
    collate_fn = TextAudioCollate()
    # pdb.set_trace()
    train_loader = DataLoader(train_dataset, num_workers=8, sampler=None, shuffle=False, batch_size=hp.batch_size, pin_memory=True,
      collate_fn=collate_fn)

    pdb.set_trace()
    for batch_idx, (text, spec, wav) in enumerate(train_loader):
        print("batch_idx" , batch_idx)
        # print("spec: ", spec)

if __name__ == '__main__':
    main()
