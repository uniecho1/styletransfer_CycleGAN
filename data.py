import torch
import os
import torchvision
import matplotlib.pylab as plt
from torchvision import transforms
from PIL import Image
import PIL


# def read_data_scenery(image_dir):
#     files = os.listdir(image_dir)
#     images = []
#     resize = torchvision.transforms.Resize(512)
#     # crop = torchvision.transforms.CenterCrop(720)
#     for file in files:
#         img = torchvision.io.read_image(
#             os.path.join(image_dir, file))
#         # feature =
#         feature = resize(img)
#         # plt.imshow(feature.permute(1, 2, 0))
#         # plt.show()
#         rect = torchvision.transforms.RandomCrop.get_params(
#             feature, (256, 256))
#         feature = torchvision.transforms.functional.crop(feature, *rect)
#         images.append(feature)
#     return images


# def read_data_scenery(image_dir):
#     transform = transforms.Compose([
#         transforms.Resize((int(1920*1.2), int(1080*1.2)),
#                           interpolation=Image.BICUBIC),
#         transforms.RandomCrop((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     for i in range(10):
#         feature = transform(img)
#         images.append(feature)
#     return images


class SceneryDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        # self.features = read_data_scenery(image_dir)
        self.images = []
        files = os.listdir(image_dir)
        for file in files:
            img = Image.open(os.path.join(image_dir, file))
            self.images.append(img)
        self.transforms = transforms.Compose([
            transforms.Resize((int(286), int(286)),
                              interpolation=Image.BICUBIC),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print("read "+str(len(self.images))+" images from "+image_dir)

    def __getitem__(self, idx):
        return self.transforms(self.images[idx]).float()

    def __len__(self):
        return len(self.images)


def load_data_scenery(batch_size):
    """返回两个图像集合的迭代器"""
    A_iter = torch.utils.data.DataLoader(
        SceneryDataset("./train/trainA"), batch_size, shuffle=True)
    B_iter = torch.utils.data.DataLoader(
        SceneryDataset("./train/trainB"), batch_size, shuffle=True)
    return A_iter, B_iter


if __name__ == "__main__":
    A_iter, B_iter = load_data_scenery(8)
    for item in B_iter:
        for i in range(item.shape[0]):
            print(item[i].shape)
            plt.imshow((item[i]*0.5+0.5).permute(1, 2, 0))
            plt.show()
