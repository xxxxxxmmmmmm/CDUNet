import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from measure import compute_measure

from mixloss import mixloss
from prep import printProgressBar
from REDCNN import RED_CNN
#from NAFNet import NAFNet
from NAFNet_wd import NAFNet
from CDUNet import CDUNet
from REDCNN_swin import REDCNN_swim
from REDCNN_SE import REDCNN_SE
from EDCNN import EDCNN

from EDCNN_loss import CompoundLoss

class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        #self.NET = RED_CNN()
        #self.NET = NAFNet()
        self.NET = CDUNet()
        #self.NET = EDCNN()
        #self.load_model(self.test_iters)

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.NET = nn.DataParallel(self.NET)
        self.NET.to(self.device)

        self.lr = args.lr
        #self.criterion = nn.L1Loss()#NAF
        #self.criterion = CompoundLoss()
        #self.criterion = nn.MSELoss()#REDCNN
        self.criterion = mixloss()

        self.optimizer = optim.Adam(self.NET.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'Net_{}iter.ckpt'.format(iter_))
        torch.save(self.NET.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'Net_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.NET.load_state_dict(state_d)
        else:
            self.NET.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()

        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        filename = '{}.png'.format(fig_name)
        # filename0 = os.path.join('E:\CCUNet_D45/', filename)
        # plt.imsave(filename0, pred,cmap='gray')
        # # filename1 = os.path.join('E:\low/', filename)
        # # plt.imsave(filename1, x,cmap='gray')
        # # filename2 = os.path.join('E:/normal/',filename)
        # # plt.imsave(filename2, y,cmap='gray')
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.NET.train(True)
            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.NET(x)
                loss = self.criterion(pred, y)
                self.NET.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))

    # 整张图像测试
    def test(self):
        del self.NET
        # load
        #self.NET = RED_CNN().to(self.device)
        #self.NET = EDCNN().to(self.device)
        self.NET = CDUNet().to(self.device)
        self.NET.eval()

        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        original_result1 = []
        original_result2 = []
        original_result3 = []
        pred_result1 = []
        pred_result2 = []
        pred_result3 = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.NET(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                #PNG
                #detach操作会从计算图中分离(detach)一个Tensor, 使这个Tensor不再参与进一步的梯度计算
                # x = x.view(shape_, shape_).cpu().detach()
                # y = y.view(shape_, shape_).cpu().detach()
                # pred = pred.view(shape_, shape_).cpu().detach()
                # x = x*255
                # y = y*255
                # pred = pred*255


                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                original_result1.append(original_result[0])
                original_result2.append(original_result[1])
                original_result3.append(original_result[2])
                pred_result1.append(pred_result[0])
                pred_result2.append(pred_result[1])
                pred_result3.append(pred_result[2])

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..\n",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader),
                                                                                            ori_ssim_avg/len(self.data_loader),
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader),
                                                                                                  pred_ssim_avg/len(self.data_loader),
                                                                                                  pred_rmse_avg/len(self.data_loader)))
            print(np.std(original_result1))

            print(np.std(original_result2))

            print(np.std(original_result3))

            print(np.std(pred_result1))

            print(np.std(pred_result2))

            print(np.std(pred_result3))

            print(pred_result1)

            print(pred_result2)

            print(pred_result3)






    # def test(self):
    #        del self.NET
    #        # load
    #        #self.NET = NAFNet().to(self.device)
    #        #self.NET = RED_CNN().to(self.device)
    #        self.NET = CDUNet().to(self.device)
    #        self.NET.eval()
    #
    #        self.load_model(self.test_iters)
    #
    #        # compute PSNR, SSIM, RMSE
    #        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    #        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    #        original_result1 = []
    #        original_result2 = []
    #        original_result3 = []
    #        pred_result1 = []
    #        pred_result2 = []
    #        pred_result3 = []
    #        l = 128
    #        w = 8
    #        n = 512 // l
    #        with torch.no_grad():
    #            for i, (x, y) in enumerate(self.data_loader):
    #                shape_ = x.shape[-1]
    #                in_data = np.pad(x, ((0, 0), (w//2, w//2), (w//2, w//2)), 'constant', constant_values=0)
    #                in_data = np.expand_dims(in_data, axis=0)
    #                in_data = torch.tensor(in_data).float().to(self.device)
    #
    #                x = x.unsqueeze(0).float().to(self.device)
    #                y = y.unsqueeze(0).float().to(self.device)
    #
    #                out_data = np.zeros([512, 512])
    #                # 分块输入并合并
    #                for x1 in range(n):
    #                    for y1 in range(n):
    #                        data = self.NET(in_data[:, :, (x1 * l):(((x1 + 1) * l) + w), (y1 * l):(((y1 + 1) * l) + w)])
    #                        data = data[:, :, (w//2):(w//2) + l, (w//2):(w//2) + l]
    #                        out_data[(x1 * l):((x1 + 1) * l), (y1 * l):((y1 + 1) * l)] = self.trunc(self.denormalize_(data.view(l, l).cpu().detach()))
    #                        # print(out_data.shape)
    #
    #
    #
    #
    #                # denormalize, truncate
    #                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
    #                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
    #                out_data = torch.tensor(out_data).float()
    #                # print(x.type)
    #                # print(y.type)
    #                # print(out_data.type)
    #
    #                data_range = self.trunc_max - self.trunc_min
    #
    #                original_result, pred_result = compute_measure(x, y, out_data, data_range)
    #                ori_psnr_avg += original_result[0]
    #                ori_ssim_avg += original_result[1]
    #                ori_rmse_avg += original_result[2]
    #                pred_psnr_avg += pred_result[0]
    #                pred_ssim_avg += pred_result[1]
    #                pred_rmse_avg += pred_result[2]
    #
    #                original_result1.append(original_result[0])
    #                original_result2.append(original_result[1])
    #                original_result3.append(original_result[2])
    #                pred_result1.append(pred_result[0])
    #                pred_result2.append(pred_result[1])
    #                pred_result3.append(pred_result[2])
    #
    #                # save result figure
    #                if self.result_fig:
    #                    self.save_fig(x, y, out_data, i, original_result, pred_result)
    #
    #                printProgressBar(i, len(self.data_loader),
    #                                 prefix="Compute measurements ..",
    #                                 suffix='Complete\n', length=25)
    #            print('\n')
    #            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader),
    #                                                                                            ori_ssim_avg/len(self.data_loader),
    #                                                                                            ori_rmse_avg/len(self.data_loader)))
    #            print('\n')
    #            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader),
    #                                                                                                  pred_ssim_avg/len(self.data_loader),
    #                                                                                                  pred_rmse_avg/len(self.data_loader)))
    #
    #            print(np.std(original_result1))
    #
    #            print(np.std(original_result2))
    #
    #            print(np.std(original_result3))
    #
    #            print(np.std(pred_result1))
    #
    #            print(np.std(pred_result2))
    #
    #            print(np.std(pred_result3))
    #
    #            print(pred_result1)
    #
    #            print(pred_result2)
    #
    #            print(pred_result3)