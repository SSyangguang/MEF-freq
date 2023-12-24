import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from kornia.losses import SSIMLoss

from option import args
from dataloader import TrainMEF, TestMEF
from net import Fusion
from loss import AMPLoss, PhaLoss


seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Train(object):
    def __init__(self):
        self.device = torch.device(args.device)
        self.num_epochs = args.epochs
        self.batch = args.batch_size
        self.lr = args.lr
        # load data and transform image to tensor and normalize
        self.train_set = TrainMEF()
        self.train_loader = data.DataLoader(self.train_set, batch_size=self.batch,
                                            shuffle=True, num_workers=0, pin_memory=True,
                                            worker_init_fn=seed_worker)

        self.n_feat = args.feat_num
        self.fusion_model = torch.nn.DataParallel(Fusion(self.n_feat), device_ids=args.devices).cuda()
        self.optimizer = AdamW(self.fusion_model.parameters(), lr=self.lr, weight_decay=args.wd)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.mse = nn.MSELoss(reduction='mean').cuda()
        self.ms_ssim = SSIMLoss(window_size=11).cuda()
        self.fre_ms_ssim = SSIMLoss(window_size=11).cuda()
        self.amp_loss = AMPLoss().cuda()
        self.pha_loss = PhaLoss().cuda()
        self.loss = []

    def train(self):
        writer = SummaryWriter(log_dir=args.log_dir, filename_suffix='train_loss')
        net = self.fusion_model

        # build folder
        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        # load pre-trained fusion model
        if os.path.exists(args.model_path + args.model):
            print('Loading pre-trained model')
            state = torch.load(args.model_path + args.model)
            net.load_state_dict(state['model'])

        for epoch in range(self.num_epochs):
            loss_total_epoch = []

            for batch, (over, under) in enumerate(self.train_loader):
                over = over.cuda()
                under = under.cuda()

                # fuse image
                fusion = net(over, under)
                self.optimizer.zero_grad(set_to_none=True)

                # calculate sim loss
                ssim_img1 = self.ms_ssim(fusion, over)
                ssim_img2 = self.ms_ssim(fusion, under)
                loss_sim = 0.5 * ssim_img1 + 0.5 * ssim_img2
                loss_sim = torch.mean(loss_sim)

                # calculate mse loss
                mse_img1 = self.mse(fusion, over)
                mse_img2 = self.mse(fusion, under)
                loss_mse = 0.5 * mse_img1 + 0.5 * mse_img2
                loss_mse = torch.mean(loss_mse)

                # calculate frequency loss
                amp_loss1 = self.amp_loss(fusion, over)
                amp_loss2 = self.amp_loss(fusion, under)
                amp_loss = 0.5 * amp_loss1 + 0.5 * amp_loss2
                amp_loss = torch.mean(amp_loss)

                pha_loss1 = self.pha_loss(fusion, over)
                pha_loss2 = self.pha_loss(fusion, under)
                pha_loss = 0.5 * pha_loss1 + 0.5 * pha_loss2
                pha_loss = torch.mean(pha_loss)

                # calculate total loss
                loss_total = loss_sim + 0.8 * loss_mse + 0.1 * amp_loss + 0.1 * pha_loss
                loss_total_epoch.append(loss_total.item())

                loss_total.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.loss.append(np.mean(loss_total_epoch))
            print('epoch: %s, loss: %s' % (epoch, np.mean(loss_total_epoch)))

            state = {
                'model': self.fusion_model.state_dict(),
                'train_loss': self.loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            torch.save(state, args.model_path + args.model)
            if epoch % 10 == 0:
                torch.save(state, args.model_path + str(epoch) + '.pth')

            writer.add_scalar('loss', np.mean(loss_total_epoch), epoch)

        fig_sim, axe_sim = plt.subplots()
        axe_sim.plot(self.loss)
        fig_sim.savefig('train_loss_curve.png')

        print('Training finished')


class TestColor(object):
    def __init__(self, batch_size=1):
        super(TestColor, self).__init__()
        self.device = torch.device(args.device)
        self.feature_num = args.feat_num

        # load data and transform image to tensor and normalize
        self.test_set = TestMEF()
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0, pin_memory=True,
                                           worker_init_fn=seed_worker)

        self.n_feat = 8
        self.fusion_model = torch.nn.DataParallel(Fusion(self.feature_num), device_ids=args.devices).cuda()
        self.fusion_state = torch.load(args.model_path + args.model, map_location='cuda:0')
        self.fusion_model.load_state_dict(self.fusion_state['model'])

        self.tau = args.tau / 255

    def test(self):
        fusion_model = self.fusion_model
        fusion_model.eval()

        for batch, (over, under, overCb, overCr, underCb, underCr, name) in enumerate(self.test_loader):
            print(name)
            over = over.cuda()
            under = under.cuda()

            outputs = fusion_model(over, under)

            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            over = over.cpu().detach().numpy()
            over = np.squeeze(over)

            under = under.cpu().detach().numpy()
            under = np.squeeze(under)

            outputs_scale = (outputs - outputs.min()) / (outputs.max() - outputs.min())

            # save color image
            fusionCb = (overCb * np.abs(overCb-self.tau) + underCb * np.abs(underCb-self.tau)) / (np.abs(overCb-self.tau) + np.abs(underCb-self.tau) + 1e-8)
            fusionCr = (overCr * np.abs(overCr-self.tau) + underCr * np.abs(underCr-self.tau)) / (np.abs(overCr-self.tau) + np.abs(underCr-self.tau) + 1e-8)
            fusionCb[(np.abs(overCb - self.tau) == 0) * (np.abs(underCb - self.tau) == 0)] = self.tau
            fusionCr[(np.abs(overCr - self.tau) == 0) * (np.abs(underCr - self.tau) == 0)] = self.tau

            color = np.stack((outputs_scale, fusionCr.squeeze(), fusionCb.squeeze()), axis=2)
            color = cv2.cvtColor(np.float32(color), cv2.COLOR_YCrCb2BGR)
            cv2.imwrite('./output/%s' % name[0].split('/')[-1], color * 255)