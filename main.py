import cv2
from whole_network import Stitch_img_network
from tools.data_prepare import DataPrepare

# 配置1： 设置要拼接的图像路径
img1_path = r'.\test_imgs\wz0101002-9.jpg'
img2_path = r'.\test_imgs\wz0101002-10.jpg'

# 配置2：设置权重路径
warp_weight_path = r'.\warp_weight\epoch040_model.pth' # 图像warp 对应的权重
compos_weight_path = r'D:\projects\RFID\exploration\img_splicing\UDIS2-inference\compos_weight\epoch050_model.pth' # 图像拼接 对应的权重


img1_tensor, img2_tensor = DataPrepare.preprocess(img1_path, img2_path)



net = Stitch_img_network(warp_weight_path=warp_weight_path, compos_weight_path=compos_weight_path)
stitched_image = net.predict(img1_tensor, img2_tensor)

# 保存图像拼接的结果
cv2.imwrite('stitched_image.jpg', stitched_image)
print('gen stitched img.....')