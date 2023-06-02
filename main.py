import data
import train
import model
import torch
import matplotlib.pylab as plt

for epoch in range(114514):
    X_iter, Y_iter = data.load_data_scenery(1)
    gen1, gen2 = model.Generator(), model.Generator()
    disc1, disc2 = model.Discriminator(), model.Discriminator()
    gen1.load_state_dict(torch.load("gen1.params"))
    gen2.load_state_dict(torch.load("gen2.params"))
    disc1.load_state_dict(torch.load("disc1.params"))
    disc2.load_state_dict(torch.load("disc2.params"))

    # for X, Y in zip(X_iter, Y_iter):
    #     X0 = X[0].detach().cpu().permute(1, 2, 0)
    #     Y0 = gen1(X)[0].detach().cpu().permute(1, 2, 0)
    #     fig, axs = plt.subplots(1, 2)
    #     axs[0].imshow(X0)
    #     axs[1].imshow(Y0)
    #     plt.show()
    #     break

    train.train(gen1, gen2, disc1, disc2, X_iter, Y_iter)
