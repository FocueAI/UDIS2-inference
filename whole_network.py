import os,torch,cv2

from warp_network import Network as WarpNetwork
from warp_network import build_model as build_warp_model
from warp_network import build_output_model as build_output_warp_model

from compos_network import Network as ComposNetwork
from compos_network import build_model as build_compos_model
from tools.data_prepare import DataPrepare
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



class Stitch_img_network:
    def __init__(self, warp_weight_path, compos_weight_path, warp_device='cuda:0', compos_device='cuda:0'):
        """
        :param warp_weight_path:     形变网络权重路径
        :param compos_weight_path:   拼接网络权重路径
        :param warp_device:          形变网络运行设备
        :param compos_device:        拼接网络运行设备
        """
        self.warp_weight_path = warp_weight_path
        self.compos_weight_path = compos_weight_path
        if torch.cuda.is_available():
            try:
                # 尝试获取 cuda:0 的设别名称
                device_name = torch.cuda.get_device_name(int(warp_device.split(':')[-1]))
                print(f'1-device_name:{device_name}')
                self.warp_device = torch.device(warp_device)
            except Exception as e:
                print(f"warp_device:{warp_device},is unavailable, using cpu")
                self.warp_device = torch.device('cpu')
            try:
                # 尝试获取 cuda:0 的设别名称
                device_name = torch.cuda.get_device_name(int(compos_device.split(':')[-1]))
                print(f'1-device_name:{device_name}')
                self.compos_device = torch.device(compos_device)
            except Exception as e:
                print(f"compos_device:{compos_device},is unavailable, using cpu")
                self.compos_device = torch.device('cpu')
        else:
            self.warp_device = torch.device('cpu')
            self.compos_device = torch.device('cpu')
        self.get_warp_network()
        self.get_compos_network()



    def get_warp_network(self):
        checkpoint = torch.load(self.warp_weight_path)
        self.warp_net = WarpNetwork()
        self.warp_net = self.warp_net.to(self.warp_device)
        self.warp_net.load_state_dict(checkpoint['model'])
        print('warp-net load model from {}!'.format(self.warp_weight_path))


    def predict(self, img1_tensor, img2_tensor):
        tmp_res = self.warp_predict(img1_tensor, img2_tensor)
        stitched_image = self.compos_predict(*tmp_res)
        return stitched_image


    def warp_predict(self, img1_tensor, img2_tensor):
        self.warp_net.eval()
        with torch.no_grad():
            batch_out = build_output_warp_model(self.warp_net, img1_tensor.to(self.warp_device), img2_tensor.to(self.warp_device))
        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']

        final_warp1 = ((final_warp1[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        final_warp2 = ((final_warp2[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1, 2, 0) * 255
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1, 2, 0) * 255
        ##################### 这些图像可以保存下来 用于查看 warp阶段的效果!!!!
        return final_warp1, final_warp2, final_warp1_mask, final_warp2_mask



    def get_compos_network(self):
        checkpoint = torch.load(self.compos_weight_path)
        self.compos_net = ComposNetwork()
        self.compos_net = self.compos_net.to(self.compos_device)
        self.compos_net.load_state_dict(checkpoint['model'])
        print('compos-net load model from {}!'.format(self.compos_weight_path))

    def compos_predict(self, final_warp1, final_warp2, final_warp1_mask, final_warp2_mask):
        final_warp1, final_warp2, final_warp1_mask, final_warp2_mask = DataPrepare.preprocess2(final_warp1, final_warp2, final_warp1_mask, final_warp2_mask)

        final_warp1 = final_warp1.to(self.compos_device)
        final_warp2 = final_warp2.to(self.compos_device)
        final_warp1_mask = final_warp1_mask.to(self.compos_device)
        final_warp2_mask = final_warp2_mask.to(self.compos_device)
        self.compos_net.eval()
        with torch.no_grad():
            batch_out = build_compos_model(self.compos_net, final_warp1, final_warp2, final_warp1_mask, final_warp2_mask)
        stitched_image = batch_out['stitched_image']
        stitched_image = ((stitched_image[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        
        return stitched_image
        










if __name__ == '__main__':
    # --------------------------step1: 数据准备 -----------------------------
    img1_path = r'.\test_imgs\wz0101002-9.jpg'
    img2_path = r'.\test_imgs\wz0101002-10.jpg'

    img1_tensor, img2_tensor = DataPrepare.preprocess(img1_path, img2_path)

    warp_weight_path = r'.\warp_weight\epoch040_model.pth' # 图像warp 对应的权重
    compos_weight_path = r'D:\projects\RFID\exploration\img_splicing\UDIS2-inference\compos_weight\epoch050_model.pth'

    net = Stitch_img_network(warp_weight_path=warp_weight_path, compos_weight_path=compos_weight_path)
    stitched_image = net.predict(img1_tensor, img2_tensor)

    # final_warp1, final_warp2, final_warp1_mask, final_warp2_mask = net.warp_predict(img1_tensor, img2_tensor)
    # stitched_image = net.compos_predict(final_warp1, final_warp2, final_warp1_mask, final_warp2_mask)

    cv2.imwrite('stitched_image.jpg', stitched_image)
    print('gen stitched img.....')



















