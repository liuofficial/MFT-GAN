import os
import torch
import argparse
from Network import network, mydatasets
from tqdm import tqdm
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
from Evaluation.Metrics import quality_mesure_fun

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MFT_GAN:
    def __init__(self, args):
        self.args = args

    @classmethod
    def load_train(cls, path, batch_size):
        datasets = mydatasets.Dataset(path)
        trainloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)
        return trainloader

    def train(self):
        train_loader = self.load_train(self.args.train_path, self.args.batch_size)
        D1_net = network.D1_net(args.hs_band).to(device)
        D2_net = network.D2_net(args.pan_band).to(device)
        G_net = network.G_net(args.hs_band, args.pan_band).to(device)
        d1_optimizer = torch.optim.AdamW(D1_net.parameters(), lr=args.lr)
        d2_optimizer = torch.optim.AdamW(D2_net.parameters(), lr=args.lr)
        g_optimizer = torch.optim.AdamW(G_net.parameters(), lr=args.lr, weight_decay=0.001)

        num1 = sum(x.numel() for x in D1_net.parameters())
        print("D1_net has {} parameters in total".format(num1))
        num2 = sum(x.numel() for x in D2_net.parameters())
        print("D2_net has {} parameters in total".format(num2))
        num3 = sum(x.numel() for x in G_net.parameters())
        print("G_net has {} parameters in total".format(num3))
        print("Total parameters : {}".format(num1 + num2 + num3))

        D1_net.train()
        D2_net.train()
        G_net.train()

        Loss1 = nn.HingeEmbeddingLoss()
        Loss2 = nn.MSELoss()
        os.makedirs(args.net_save_path, exist_ok=True)

        for epoch in range(args.max_epoch):

            for num, data1 in tqdm(enumerate(train_loader), total=len(train_loader)):
                Y = data1['Y'].to(device)
                Z = data1['Z'].to(device)
                X1 = F.interpolate(Y, scale_factor=args.ratio, mode='bicubic', align_corners=False)

                real_labels = torch.full((X1.size(0), 1), 1.0, requires_grad=False).to(device)
                fake_labels = torch.full((X1.size(0), 1), -1.0, requires_grad=False).to(device)

                # Train sprctral discriminator
                D1_net.zero_grad()
                fake_hrhs = G_net(Y, Z)
                fake_pan = torch.mean(fake_hrhs.detach(), dim=1).unsqueeze(1)
                output_fake_d1 = D1_net(fake_hrhs.detach())
                loss1_1 = Loss1(output_fake_d1, fake_labels)
                output_real_d1 = D1_net(X1)
                loss1_2 = Loss1(output_real_d1, real_labels)

                d2_loss = loss1_1 + loss1_2
                d2_loss.backward()
                d2_optimizer.step()

                # Train spatial discriminator
                D2_net.zero_grad()
                output_fake_d2 = D2_net(fake_pan)
                loss2_1 = Loss1(output_fake_d2, fake_labels)
                output_real_d1 = D2_net(Z)
                loss2_2 = Loss1(output_real_d1, real_labels)

                d1_loss = loss2_1 + loss2_2
                d1_loss.backward()
                d1_optimizer.step()

                # Train generator
                G_net.zero_grad()
                fake_hrhs = G_net(Y, Z)
                fake_pan = torch.mean(fake_hrhs, dim=1).unsqueeze(1)

                out_hrhs = D1_net(fake_hrhs)
                ad1_loss = Loss1(out_hrhs, real_labels)

                out_pan = D2_net(fake_pan)
                ad2_loss = Loss1(out_pan, real_labels)

                g_spectrum_loss = 1.0 - torch.mean(F.cosine_similarity(fake_hrhs, X1, dim=1))
                g_spatial_loss = Loss2(fake_pan, Z)
                gan_loss = ad2_loss + ad1_loss

                # Loss Item
                g_loss = g_spatial_loss + g_spectrum_loss + 0.001 * gan_loss
                g_loss.backward()
                g_optimizer.step()

            if ((epoch + 1) % 50 == 0):
                torch.save(G_net.state_dict(), args.net_save_path + 'generator_' + str(epoch + 1) + '.pkl')
                torch.save(D1_net.state_dict(), args.net_save_path + 'discriminator1_' + str(epoch + 1) + '.pkl')
                torch.save(D2_net.state_dict(), args.net_save_path + 'discriminator2_' + str(epoch + 1) + '.pkl')
                print('Models save to ./Model/generator.pkl & ./Model/discriminator.pkl ')

    @classmethod
    def test_piece(self, data_flag, stride=None):
        net = network.G_net(hs_band=args.hs_band, pan_band=args.pan_band)
        net_dict = torch.load(args.net_save_path + 'generator_300.pkl', weights_only=True)
        net.load_state_dict(net_dict)
        net.eval()
        net.to(device)

        if data_flag == 'CAVE':
            # CAVE
            num_start = 21
            num_end = 32

        if data_flag == 'CHK':
            # CHK
            num_start = 46
            num_end = 64

        elif data_flag == 'BST':
            # BST
            num_start = 15
            num_end = 20

        elif data_flag == 'LA':
            # LA
            num_start = 1
            num_end = 1

        os.makedirs(args.results_path, exist_ok=True)
        if stride is None:
            stride = 16
        for i in range(num_start, num_end + 1):
            mat = sio.loadmat(args.data_path + '%d.mat' % i)
            tY = mat['Y']
            tZ = mat['P']
            output = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
            num_sum = np.zeros([tZ.shape[0], tZ.shape[1], tY.shape[2]])
            for x in range(0, tZ.shape[0] - args.test_patch_size + 1, stride):
                for y in range(0, tZ.shape[1] - args.test_patch_size + 1, stride):
                    end_x = x + args.test_patch_size
                    if end_x + stride > tZ.shape[0]:
                        end_x = tZ.shape[0]
                    end_y = y + args.test_patch_size
                    if end_y + stride > tZ.shape[1]:
                        end_y = tZ.shape[1]
                    itY = tY[x // args.ratio:end_x // args.ratio, y // args.ratio:end_y // args.ratio, :]
                    itZ = tZ[x:end_x, y:end_y, :]
                    lrhs = np.transpose(itY, (2, 0, 1))
                    lrhs = torch.from_numpy(lrhs).type(torch.FloatTensor).unsqueeze(0).to(device)
                    ms = np.transpose(itZ, (2, 0, 1))
                    ms = torch.from_numpy(ms).type(torch.FloatTensor).unsqueeze(0).to(device)
                    tmp = net(lrhs, ms)
                    tmp = tmp.cpu().squeeze().detach().numpy()
                    tmp = np.transpose(tmp, (1, 2, 0))
                    output[x:end_x, y:end_y, :] += tmp
                    num_sum[x:end_x, y:end_y, :] += 1

            output = output / num_sum
            output[output < 0] = 0.0
            output[output > 1] = 1.0
            sio.savemat(args.results_path + str(i) + '.mat', {'hs': output})
            print('%d has finished' % i)
        quality_mesure_fun(args.results_path, args.data_path, num_start, num_end)

    @classmethod
    def Test(self, data_flag):
        net = network.G_net(hs_band=args.hs_band, pan_band=args.pan_band)
        net_dict = torch.load(args.net_save_path + 'generator_300.pkl', weights_only=True)
        net.load_state_dict(net_dict)
        net.eval()

        if data_flag == 'CAVE':
            # CAVE
            num_start = 21
            num_end = 32

        if data_flag == 'CHK':
            # CHK
            num_start = 46
            num_end = 64

        elif data_flag == 'BST':
            # BST
            num_start = 15
            num_end = 20

        elif data_flag == 'LA':
            # LA
            num_start = 1
            num_end = 1

        os.makedirs(args.results_path, exist_ok=True)
        for j in range(num_start, num_end + 1):
            path = args.data_path + str(j) + '.mat'
            data = sio.loadmat(path)
            ms = data['P']
            lrhs = data['Y']
            lrhs = np.transpose(lrhs, (2, 0, 1))
            lrhs = torch.from_numpy(lrhs).type(torch.FloatTensor).unsqueeze(0)
            ms = np.transpose(ms, (2, 0, 1))
            ms = torch.from_numpy(ms).type(torch.FloatTensor).unsqueeze(0)
            out = net(lrhs, ms)
            out1 = out.cpu().squeeze().detach().numpy()
            out1 = np.transpose(out1, (1, 2, 0))
            out1[out1 < 0] = 0.0
            out1[out1 > 1] = 1.0
            sio.savemat(args.results_path + str(j) + '.mat', {'hs': out1})
            print('%d has finished' % j)
        quality_mesure_fun(args.results_path, args.data_path, num_start, num_end)


if __name__ == '__main__':
    ############## arguments
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--max_epoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument("--ratio", type=int, default=4, help="ms_band")
    parser.add_argument("--lr", type=int, default=0.0001, help="learning_rate")
    parser.add_argument("--test_patch_size", type=int, default=128)

    data_flag = 'CAVE'
    if data_flag == 'CAVE':
        # CAVE
        parser.add_argument("--hs_band", type=int, default=31, help="hs_band")
        parser.add_argument("--pan_band", type=int, default=1, help="ms_band")
        parser.add_argument("--train_path", type=str, default=r'XXX/Sample_data/CAVE/CAVE_patch_32/train/',
                            help="train_path")
        parser.add_argument("--data_path", type=str, default=r'XXX/Sample_data/CAVE/CAVEMAT_r4/')
        parser.add_argument("--net_save_path", type=str, default='./Models/CAVE/')
        parser.add_argument("--results_path", type=str, default='./Results/CAVE_test_HRHS/')

    elif data_flag == 'CHK':
        # CHK
        parser.add_argument("--hs_band", type=int, default=128, help="hs_band")
        parser.add_argument("--pan_band", type=int, default=1, help="ms_band")
        parser.add_argument("--train_path", type=str,
                            default=r'XXX/Sample_data/Chikusei/CHK_patch_32/train/', help="train_path")
        parser.add_argument("--net_save_path", type=str, default='./Models/CHK/')
        parser.add_argument("--results_path", type=str, default='./Results/CHK_test_HRHS/')
        parser.add_argument("--data_path", type=str,
                            default=r'XXX/Sample_data/Chikusei/CHKMAT_r4/')

    elif data_flag == 'BST':
        # BST
        parser.add_argument("--hs_band", type=int, default=145, help="hs_band")
        parser.add_argument("--pan_band", type=int, default=1, help="ms_band")
        parser.add_argument("--train_path", type=str, default=r'XXX/Sample_data/Bostwana/BST_patch_32/train/', help="train_path")
        parser.add_argument("--data_path", type=str, default=r'XXX/Sample_data/Bostwana/BSTMAT_r4/')
        parser.add_argument("--net_save_path", type=str, default='./Models/BST/')
        parser.add_argument("--results_path", type=str, default='./Results/BST_test_HRHS/')


    elif data_flag == 'LA':
        # LA
        parser.add_argument("--hs_band", type=int, default=145, help="hs_band")
        parser.add_argument("--pan_band", type=int, default=1, help="ms_band")
        parser.add_argument("--train_path", type=str,default=r'XXX/Sample_data/Los_Angeles/LA_patch_48/train/', help="train_path")
        parser.add_argument("--net_save_path", type=str, default='./Models/LA/')
        parser.add_argument("--results_path", type=str, default='./Results/LA_test_HRHS/')
        parser.add_argument("--data_path", type=str,default=r'XXX/Sample_data/Los_Angeles/LAMAT_r4/')

    args = parser.parse_args()
    torch.cuda.empty_cache()
    mft_gan = MFT_GAN(args)
    is_train = True
    if is_train:
        mft_gan.train()
    else:
        mft_gan.test_piece(data_flag)
        # or
        # mft_gan.Test(data_flag)