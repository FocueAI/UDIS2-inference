import os,torch,cv2

from warp_network import Network as WarpNetwork
from warp_network import build_model as build_warp_model
from warp_network import build_output_model as build_output_warp_model

from compos_network import Network as ComposNetwork
from compos_network import build_model as build_compos_model
from tools.data_prepare import DataPrepare
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



if __name__ == '__main__':
    # --------------------------step1: 数据准备 -----------------------------
    img1_path = r'.\test_imgs\wz0101002-9.jpg'
    img2_path = r'.\test_imgs\wz0101002-10.jpg'

    img1_tensor, img2_tensor = DataPrepare.preprocess(img1_path, img2_path)

    # ------------------------ step2: 网络准备 -------------------------------
    # warp 网络
    warp_weight_path = r'.\warp_weight\epoch040_model.pth' # 图像warp 对应的权重
    checkpoint = torch.load(warp_weight_path)
    warp_net = WarpNetwork()
    if torch.cuda.is_available():
        warp_on_device = torch.device('cuda:0')
    else:
        warp_on_device = torch.device('cpu')
    # warp_on_device = torch.device('cpu')
    warp_net = warp_net.to(warp_on_device)
    warp_net.load_state_dict(checkpoint['model'])
    print('warp-net load model from {}!'.format(warp_weight_path))
    # composition 网络
    compos_weight_path = r'D:\projects\RFID\exploration\img_splicing\UDIS2-inference\compos_weight\epoch050_model.pth'

    checkpoint = torch.load(compos_weight_path)
    compos_net = ComposNetwork()
    if torch.cuda.is_available():
        compos_on_device = torch.device('cuda:0')
    else:
        compos_on_device = torch.device('cpu')
    # compos_on_device = torch.device('cpu')
    compos_net = compos_net.to(compos_on_device)
    compos_net.load_state_dict(checkpoint['model'])
    print('compos-net load model from {}!'.format(compos_weight_path))
    # ---------------------------step3: 开始推理 ---------------------------------
    img1_tensor = img1_tensor.to(warp_on_device)
    img2_tensor = img2_tensor.to(warp_on_device)
    warp_net.eval()
    with torch.no_grad():
        batch_out = build_output_warp_model(warp_net, img1_tensor, img2_tensor)
    final_warp1 = batch_out['final_warp1']
    final_warp1_mask = batch_out['final_warp1_mask']
    final_warp2 = batch_out['final_warp2']
    final_warp2_mask = batch_out['final_warp2_mask']
    # --------------------------------------------------
    final_warp1 = ((final_warp1[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    final_warp2 = ((final_warp2[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1, 2, 0) * 255
    final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1, 2, 0) * 255


    final_warp1, final_warp2, final_warp1_mask, final_warp2_mask = DataPrepare.preprocess2(final_warp1, final_warp2, final_warp1_mask, final_warp2_mask)

    final_warp1 = final_warp1.to(compos_on_device)
    final_warp2 = final_warp2.to(compos_on_device)
    final_warp1_mask = final_warp1_mask.to(compos_on_device)
    final_warp2_mask = final_warp2_mask.to(compos_on_device)
    compos_net.eval()
    with torch.no_grad():
        batch_out = build_compos_model(compos_net, final_warp1, final_warp2, final_warp1_mask, final_warp2_mask)
    stitched_image = batch_out['stitched_image']
    stitched_image = ((stitched_image[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    cv2.imwrite('stitched_image.jpg', stitched_image)




















