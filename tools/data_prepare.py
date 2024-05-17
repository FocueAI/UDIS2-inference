import cv2, torch
import numpy as np

class DataPrepare:

    img_width = 2000#2000# 2377 # 512
    img_height = 1200#1500# 1839 # 512

    @staticmethod
    def preprocess(img1_path, img2_path):

        # prepare img1
        input1 = cv2.imread(img1_path)
        input1 = cv2.resize(input1, (DataPrepare.img_width, DataPrepare.img_height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        # prepare img2
        input2 = cv2.imread(img2_path)
        input2 = cv2.resize(input2, (DataPrepare.img_width, DataPrepare.img_height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return input1_tensor.unsqueeze(0), input2_tensor.unsqueeze(0)

    @staticmethod
    def preprocess2(warp1_img, warp2_img, mask1_img, mask2_img):
        """
        :param warp1_img:
        :param warp2_img:
        :param mask1_img:
        :param mask2_img:
        :return:
        """
        # convert warp1
        warp1 = warp1_img.astype(np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # convert warp2
        warp2 = warp2_img.astype(np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # convert mask1
        mask1 = mask1_img.astype(np.float32)
        mask1 = (mask1 / 127.5) - 1.0
        mask1 = np.transpose(mask1, [2, 0, 1])

       # convert mask2
        mask2 = mask2_img.astype(np.float32)
        mask2 = (mask2 / 127.5) - 1.0
        mask2 = np.transpose(mask2, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1).unsqueeze(0)
        warp2_tensor = torch.tensor(warp2).unsqueeze(0)
        mask1_tensor = torch.tensor(mask1).unsqueeze(0)
        mask2_tensor = torch.tensor(mask2).unsqueeze(0)

        return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor












