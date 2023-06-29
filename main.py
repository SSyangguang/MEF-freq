import random
import torch
import numpy as np

from model import Train, Test, TestColor
from option import args

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)


def set_seed():
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    train_trans = Train()
    train_trans.train()


def test():
    test = TestColor()
    test.test()


if __name__ == '__main__':
    set_seed()
    train()
