import torch
from torch import nn
import utils
import data
import matplotlib.pylab as plt
import random


class ImgBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, img):
        if (len(self.data) < self.max_size):
            self.data.append(img)
            return img
        else:
            if (random.uniform(0, 1) > 0.5):
                return img
            else:
                i = random.randint(0, self.max_size-1)
                res = self.data[i]
                self.data[i] = img
                return res


def train(gen1, gen2, disc1, disc2, X_iter, Y_iter,
          num_epoch=200,
          gan_weight=1, cycle_weight=10, identity_weight=2,
          learning_rate=0.0002, weight_decay=0,
          device=utils.try_gpu()):
    gen1 = gen1.to(device)
    gen2 = gen2.to(device)
    disc1 = disc1.to(device)
    disc2 = disc2.to(device)
    gen1buffer = ImgBuffer(50)
    gen2buffer = ImgBuffer(50)
    gen1_optimizer = torch.optim.Adam(
        gen1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    gen2_optimizer = torch.optim.Adam(
        gen2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    disc1_optimizer = torch.optim.Adam(
        disc1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    disc2_optimizer = torch.optim.Adam(
        disc2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    target_real = torch.tensor([[1.]], requires_grad=False).to(device)
    target_fake = torch.tensor([[0.]], requires_grad=False).to(device)
    for epoch in range(num_epoch):
        for i, (X, Y) in enumerate(zip(X_iter, Y_iter)):
            X = X.to(device)
            Y = Y.to(device)
            mseloss = torch.nn.MSELoss()
            for k in range(5):
                loss1 = (mseloss(disc1(X), target_real) +
                         mseloss(disc1(gen2buffer.push_and_pop(gen2(Y).detach())), target_fake))*0.5
                # loss1 = loss1.float()
                disc1_optimizer.zero_grad()
                loss1.backward()
                disc1_optimizer.step()

                loss2 = (mseloss(disc2(Y), target_real) +
                         mseloss(disc2(gen1buffer.push_and_pop(gen1(X).detach())), target_fake))*0.5
                disc2_optimizer.zero_grad()
                loss2.backward()
                disc2_optimizer.step()

            L1loss = torch.nn.L1Loss()

            gen1_identity_loss = L1loss(Y, gen1(Y))
            gen1_gan_loss = mseloss(disc2(gen1(X)), target_real)
            gen1_cycle_loss = L1loss(gen2(gen1(X)), X)
            gen1_loss = gen1_identity_loss*identity_weight + \
                gen1_cycle_loss*cycle_weight+gen1_gan_loss*gan_weight
            gen1_optimizer.zero_grad()
            gen1_loss.backward()
            gen1_optimizer.step()

            gen2_identity_loss = L1loss(X, gen2(X))
            gen2_gan_loss = mseloss(disc1(gen2(Y)), target_real)
            gen2_cycle_loss = L1loss(gen1(gen2(Y)), Y)
            gen2_loss = gen2_identity_loss*identity_weight + \
                gen2_cycle_loss*cycle_weight+gen2_gan_loss*gan_weight
            gen2_optimizer.zero_grad()
            gen2_loss.backward()
            gen2_optimizer.step()

            print(
                f"epoch = epoch {epoch}, disc1_loss = {loss1}, disc2_loss = {loss2}, gen1_loss = {gen1_loss}, gen2_loss = {gen2_loss}")

        if (epoch+1) % 5 == 0:
            # X0 = X[0].detach().cpu().permute(1, 2, 0)
            # Y0 = gen1(X)[0].detach().cpu().permute(1, 2, 0)
            # fig, axs = plt.subplots(1, 2)
            # X0 = X0*0.5+0.5
            # Y0 = Y0*0.5+0.5
            # axs[0].imshow(X0)
            # axs[1].imshow(Y0)
            # plt.show()
            torch.save(gen1.state_dict(), "gen1.params")
            torch.save(gen2.state_dict(), "gen2.params")
            torch.save(disc1.state_dict(), "disc1.params")
            torch.save(disc2.state_dict(), "disc2.params")
