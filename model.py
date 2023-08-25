import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import wandb
# from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from kornia.losses import MS_SSIMLoss, SSIMLoss

from option import args
from dataloader import TrainMEF, TestMEF
from net import Fusion, DenseNet, vgg16
from loss import ssim, ms_ssim, SSIM, MS_SSIM, AMPLoss, PhaLoss

from GPPNN_models.GPPNN_freq import GPPNN

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
        # wandb.login()  # wandb api key
        # runs = wandb.init(project='freq-fusion', mode='online')
        # self.runs = runs

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
        # self.fusion_model = Fusion(self.n_feat).to(self.device)
        self.fusion_model = torch.nn.DataParallel(Fusion(self.n_feat), device_ids=args.devices).cuda()
        self.optimizer = AdamW(self.fusion_model.parameters(), lr=self.lr, weight_decay=args.wd)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.vgg = vgg16().cuda()
        self.mse = nn.MSELoss(reduction='mean').cuda()
        self.ms_ssim = SSIMLoss(window_size=11).cuda()  # 0.8
        self.fre_ms_ssim = SSIMLoss(window_size=11).cuda()
        # self.ms_ssim = MS_SSIMLoss(alpha=0.1).cuda()    # 0.8
        # self.fre_ms_ssim = MS_SSIMLoss(alpha=0.5).cuda()
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
                # t = tqdm(self.train_loader, disable=False, total=len(self.train_loader), ncols=120)

                over = over.cuda()
                under = under.cuda()

                # fuse image
                fusion, fre1, fre2, fre3, fre4 = net(over, under)
                fusion = fusion.cuda()
                fre1 = fre1.cuda()
                fre2 = fre2.cuda()
                fre3 = fre3.cuda()
                fre4 = fre4.cuda()
                # fre_output = fre_output.cuda()
                # print(fusion)

                self.optimizer.zero_grad(set_to_none=True)

                # # freq branch loss
                # fre_loss1 = self.fre_ms_ssim(fre_output, over)
                # fre_loss2 = self.fre_ms_ssim(fre_output, under)
                # loss_fre = 0.5 * fre_loss1 + 0.5 * fre_loss2
                # loss_fre = torch.mean(loss_fre)

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

                # calculate per loss
                # per_over = self.vgg(torch.cat((over, over, over), dim=1))
                # per_under = self.vgg(torch.cat((under, under, under), dim=1))
                # per_fusion = self.vgg(torch.cat((fusion, fusion, fusion), dim=1))
                # per_loss1 = self.mse(per_fusion[4], per_over[4])
                # per_loss2 = self.mse(per_fusion[4], per_under[4])
                # loss_per = 0.5 * per_loss1 + 0.5 * per_loss2
                # loss_per = torch.mean(loss_per)

                # calculate fre branch loss
                # fre_loss1 = self.ms_ssim(fre1, over) + self.ms_ssim(fre1, under) + 0.2 * (self.mse(fre1, over) + self.mse(fre1, under))
                # fre_loss2 = self.ms_ssim(fre2, over) + self.ms_ssim(fre2, under) + 0.2 * (
                #             self.mse(fre2, over) + self.mse(fre2, under))
                # fre_loss3 = self.ms_ssim(fre3, over) + self.ms_ssim(fre3, under) + 0.2 * (
                #             self.mse(fre3, over) + self.mse(fre3, under))
                # fre_loss4 = self.ms_ssim(fre4, over) + self.ms_ssim(fre4, under) + 0.2 * (
                #             self.mse(fre4, over) + self.mse(fre4, under))

                fre_loss1 = self.amp_loss(fre1, over) + self.amp_loss(fre1, under) + 0.8 * (
                            self.pha_loss(fre1, over) + self.pha_loss(fre1, under))
                fre_loss2 = self.amp_loss(fre2, over) + self.amp_loss(fre2, under) + 0.8 * (
                        self.pha_loss(fre2, over) + self.pha_loss(fre2, under))
                fre_loss3 = self.amp_loss(fre3, over) + self.amp_loss(fre3, under) + 0.8 * (
                        self.pha_loss(fre3, over) + self.pha_loss(fre3, under))
                fre_loss4 = self.amp_loss(fre4, over) + self.amp_loss(fre4, under) + 0.8 * (
                        self.pha_loss(fre4, over) + self.pha_loss(fre4, under))

                fre_bra_loss = fre_loss1 + fre_loss2 + fre_loss3 + fre_loss4

                # calculate total loss
                loss_total = loss_sim + loss_mse + 2 * fre_bra_loss # amp_loss本来是0.1
                # loss_total = loss_sim + 0.8 * loss_mse + 0.1 * amp_loss + 0.1 * pha_loss + 2 * fre_bra_loss  # amp_loss本来是0.1
                loss_total_epoch.append(loss_total.item())

                loss_total.backward()
                self.optimizer.step()

            self.scheduler.step()
            # self.scheduler.step()
            self.loss.append(np.mean(loss_total_epoch))
            print('epoch: %s, loss: %s' % (epoch, np.mean(loss_total_epoch)))

            loss_state = {
                'train_loss': self.loss
            }
            # wandb.log(loss_state)

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


class Test(object):
    def __init__(self, batch_size=1):
        super(Test, self).__init__()
        self.device = torch.device(args.device)
        self.feature_num = args.feature_num

        # load data and transform image to tensor and normalize
        self.test_set = TestMEF()
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0, pin_memory=True,
                                           worker_init_fn=seed_worker)

        self.fusion_model = torch.nn.DataParallel(Fusion(self.n_feat), device_ids=args.devices).cuda()
        self.fusion_state = torch.load(args.model_path + args.model, map_location=args.device)
        self.fusion_model.load_state_dict(self.fusion_state['model'])

    def test(self):
        fusion_model = self.fusion_model
        fusion_model.eval()

        for batch, (ir_img, vis_img, ir_name, vis_name) in enumerate(self.test_loader):
            ir_img = ir_img.cuda()
            vis_img = vis_img.cuda()

            outputs, _, _, _ = fusion_model(ir_img, vis_img)

            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            ir_img = ir_img.cpu().detach().numpy()
            ir_img = np.squeeze(ir_img)

            vis_img = vis_img.cpu().detach().numpy()
            vis_img = np.squeeze(vis_img)

            outputs_scale = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            # outputs_scale = (outputs - outputs.mean()) / outputs.std()
            outputs_scale = (outputs_scale * 255).astype(np.int)

            if not os.path.exists(args.result_path):
                os.mkdir(args.result_path)
            cv2.imwrite('%s/%s.png' % (args.result_path, ir_name[0]), outputs_scale)

            # outputs = (outputs * 255).astype(np.int)
            # cv2.imwrite('test.jpg', outputs)


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

            # outputs = self.block_fusion(over.squeeze(), under.squeeze(), block_size=args.block_size)
            outputs, _, _, _, _ = fusion_model(over, under)

            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            over = over.cpu().detach().numpy()
            over = np.squeeze(over)

            under = under.cpu().detach().numpy()
            under = np.squeeze(under)

            # # save gray fusion image
            # outputs_scale = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            # # outputs_scale = (outputs - outputs.mean()) / outputs.std()
            # outputs_scale = (outputs_scale * 255).astype(np.int)
            # # cv2.imwrite('E:\project\code\multitask2023/kaist-simplified/fusion/%s.jpg' % ir_name[0], outputs_scale)

            # save color image
            fusionCb = (overCb * np.abs(overCb-self.tau) + underCb * np.abs(underCb-self.tau)) / (np.abs(overCb-self.tau) + np.abs(underCb-self.tau) + 1e-8)
            fusionCr = (overCr * np.abs(overCr-self.tau) + underCr * np.abs(underCr-self.tau)) / (np.abs(overCr-self.tau) + np.abs(underCr-self.tau) + 1e-8)
            fusionCb[(np.abs(overCb - self.tau) == 0) * (np.abs(underCb - self.tau) == 0)] = self.tau
            fusionCr[(np.abs(overCr - self.tau) == 0) * (np.abs(underCr - self.tau) == 0)] = self.tau

            color = np.stack((outputs, fusionCr.squeeze(), fusionCb.squeeze()), axis=2)
            color = cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite('./output/%s' % name[0].split('/')[-1], color * 255)

    def get_block(self, img, block_size=args.block_size):
        '''
        The original image is cut into blocks according to block_size
        output: blocks [blocks_num, block_size, block_size]
        '''
        m, n = img.shape
        blocks = torch.ones((block_size, block_size)).cuda()
        blocks = torch.unsqueeze(blocks, dim=0)

        img_pad = F.pad(img, (0, block_size - n % block_size, 0, block_size - m % block_size), 'constant')  # mirror padding
        # img_pad = np.pad(img, ((0, 256 - m % block_size), (0, 256 - n % block_size)), 'reflect')  # mirror padding
        m_block = int(np.ceil(m / block_size))  # Calculate the total number of blocks
        n_block = int(np.ceil(n / block_size))  # Calculate the total number of blocks

        # cutting
        for i in range(0, m_block):
            for j in range(0, n_block):
                block = torch.unsqueeze(img_pad[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size], dim=0)
                blocks = torch.cat((blocks, block), dim=0)
        blocks = blocks[1:, :, :]

        return blocks

    def fuse(self, img1, img2, block_size=args.block_size):
        '''
        block fusion
        '''
        block_num = img1.shape[0]
        final_fusion = torch.ones(block_num, block_size, block_size)
        # final_fusion = np.zeros_like(img1)

        for i in range(block_num):
            img1_inblock = torch.unsqueeze(img1[i:i+1, :, :], dim=0)
            img2_inblock = torch.unsqueeze(img2[i:i+1, :, :], dim=0)

            img_fusion = self.fusion_model(img1_inblock, img2_inblock)

            # note that no normalization should be used in different block fusion
            # img_fusion = MaxMinNormalization(img_fusion[0], torch.max(img_fusion[0]), torch.min(img_fusion[0]))
            # img_fusion = img_fusion.numpy()

            final_fusion[i, :, :] = torch.squeeze(img_fusion, dim=0)

        return final_fusion

    def block_to_img(self, block_img, m, n):
        '''
        Enter the fused block and restore it to the original image size.
        '''
        block_size = block_img.shape[2]
        m_block = int(np.ceil(m / block_size))
        n_block = int(np.ceil(n / block_size))
        fused_full_img_wpad = torch.ones(m_block * block_size, n_block * block_size)  # Image size after padding
        for i in range(0, m_block):
            for j in range(0, n_block):
                fused_full_img_wpad[i * block_size: (i + 1) * block_size,
                j * block_size: (j + 1) * block_size] = block_img[i * n_block + j, :, :]
        fused_full_img = fused_full_img_wpad[:m, :n]  # image with original size
        return fused_full_img

    def block_fusion(self, img1, img2, block_size=256):
        '''
        Input img1, img2, slice block according to block_size and fuse, output result
        '''
        # blocks_img大小[blocks_num, block_size, block_size, 3]
        blocks_img1 = self.get_block(img1, block_size=block_size)
        blocks_img2 = self.get_block(img2, block_size=block_size)
        print('img1', blocks_img1.shape)
        print('img2', blocks_img2.shape)

        # fusion
        fused_block_img1 = self.fuse(blocks_img1, blocks_img2)

        # block restore to orginal size
        fused_img = self.block_to_img(fused_block_img1, img1.shape[0], img1.shape[1])

        return fused_img
