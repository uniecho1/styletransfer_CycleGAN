import cv2
import os

image_dir = "./train/opraw"
files = os.listdir(image_dir)
for file in files:
    img = cv2.imread(
        os.path.join(image_dir, file))
    # print(img.shape)
    cropped1 = img[0:898, 0:1920]
    cropped2 = img[0:1080, 0:1644]
    # cv2.imshow("fuck", cropped)
    cv2.imwrite(os.path.join("./train/trainC", "1"+file), cropped1)
    cv2.imwrite(os.path.join("./train/trainC", "2"+file), cropped2)
