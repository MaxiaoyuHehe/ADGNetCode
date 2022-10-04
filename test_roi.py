from PIL import Image
import torchvision
import torch
import models
import cv2
import numpy as np
import os
import scipy.io as scio

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def open_img(path):
    mytransform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])
    img = pil_loader(path)
    sample = mytransform(img)
    H, W = sample.size(1), sample.size(2)
    newH, newW = H, W
    if H % 32 != 0:
        newH = ((H // 32) + 1) * 32
    if W % 32 != 0:
        newW = ((W // 32) + 1) * 32
    newSample = torch.zeros((3, newH, newW))
    newSample[:, (newH - H) // 2:(newH - H) // 2 + H, (newW - W) // 2:(newW - W) // 2 + W] = sample

    visual_img = cv2.imread(path)
    new_visual_img = np.zeros((newH, newW, 3), dtype=np.uint8)
    new_visual_img[(newH - H) // 2:(newH - H) // 2 + H, (newW - W) // 2:(newW - W) // 2 + W, :] = visual_img

    return torch.unsqueeze(newSample, dim=0), new_visual_img


def test_roi(path, mymod):
    img = open_img(path).cuda()
    out = mymod(img)
    return out


def test_img(in_path, out_path):
    mymod = models.RoiADGNet().cuda()
    mymod.train(False)
    mymod.eval()
    img, _ = open_img(in_path)
    out = mymod(img.cuda())
    q1 = torch.squeeze(out['Q1']).detach().cpu().numpy()
    q1[q1 < 0] = 0
    q1[q1 > 1] = 1

    q2 = torch.squeeze(out['Q2']).detach().cpu().numpy()
    q2[q2 < 0] = 0
    q2[q2 > 1] = 1
    visual_img = cv2.imread(in_path)
    H, W = visual_img.shape[0], visual_img.shape[1]
    cam_img01 = show_cam_on_image(np.float32(visual_img) / 255.0, q1, use_rgb=False)
    cam_img02 = show_cam_on_image(np.float32(visual_img) / 255.0, q2, use_rgb=False)
    out_img = np.zeros((H * 3, W, 3), dtype=np.uint8)
    out_img[:H, :, :] = visual_img
    out_img[H:H * 2, :, :] = cam_img01
    out_img[H * 2:, :, :] = cam_img02
    att1 = torch.squeeze(out['Att1']).detach().cpu().numpy()
    att2 = torch.squeeze(out['Att2']).detach().cpu().numpy()
    lcft = torch.squeeze(out['LcFt']).detach().cpu().numpy()
    scio.savemat('att1.mat',{'att1': att1})
    scio.savemat('att2.mat', {'att2': att2})
    scio.savemat('lcft.mat', {'lcft': lcft})

    cv2.imwrite(out_path, out_img)


def test_video(in_folder, out_folder, len):
    mymod = models.RoiADGNet().cuda()
    mymod.train(False)
    mymod.eval()
    for i in range(len):
        in_path = '%s-%03d.png' % (in_folder, i + 1)
        img, visual_img = open_img(in_path)
        out = mymod(img.cuda())
        q1 = torch.squeeze(out['Q1']).detach().cpu().numpy()
        q1[q1 < 0] = 0
        q1[q1 > 1] = 1

        q2 = torch.squeeze(out['Q2']).detach().cpu().numpy()
        q2[q2 < 0] = 0
        q2[q2 > 1] = 1
        H, W = visual_img.shape[0], visual_img.shape[1]
        cam_img01 = show_cam_on_image(np.float32(visual_img) / 255.0, q1, use_rgb=False)
        cam_img02 = show_cam_on_image(np.float32(visual_img) / 255.0, q2, use_rgb=False)
        out_img = np.zeros((H * 3, W, 3), dtype=np.uint8)
        out_img[:H, :, :] = visual_img
        out_img[H:H * 2, :, :] = cam_img01
        out_img[H * 2:, :, :] = cam_img02
        cv2.putText(out_img, 'BitRate:10M+', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (30, 200, 30), 10)
        cv2.imwrite('%s-%03d.png' % (out_folder, i + 1), out_img)
        print(i)


def merge():
    cnt = 0
    for i in range(200):
        imgLeft = cv2.imread('E:\\TempWork\\aq\\aq-%03d.png' % (i + 301))
        imgRight = cv2.imread('E:\\TempWork\\adq\\adq-%03d.png' % (i + 301))
        imgsMerge = np.zeros((1088 * 3, 1920 * 2, 3), dtype=np.uint8)
        imgsMerge[:, :1920, :] = imgLeft
        imgsMerge[:, 1920:, :] = imgRight
        cv2.imwrite('E:\\TempWork\\show\\show-%04d.png' % cnt, imgsMerge)
        cnt += 1
        print(cnt)

    for i in range(200):
        imgLeft = cv2.imread('E:\\TempWork\\bq\\bq-%03d.png' % (i + 301))
        imgRight = cv2.imread('E:\\TempWork\\bdq\\bdq-%03d.png' % (i + 301))
        imgsMerge = np.zeros((1088 * 3, 1920 * 2, 3), dtype=np.uint8)
        imgsMerge[:, :1920, :] = imgLeft
        imgsMerge[:, 1920:, :] = imgRight
        cv2.imwrite('E:\\TempWork\\show\\show-%04d.png' % cnt, imgsMerge)
        cnt += 1
        print(cnt)

    for i in range(200):
        imgLeft = cv2.imread('E:\\TempWork\\cq\\cq-%03d.png' % (i + 301))
        imgRight = cv2.imread('E:\\TempWork\\cdq\\cdq-%03d.png' % (i + 301))
        imgsMerge = np.zeros((1088 * 3, 1920 * 2, 3), dtype=np.uint8)
        imgsMerge[:, :1920, :] = imgLeft
        imgsMerge[:, 1920:, :] = imgRight
        cv2.imwrite('E:\\TempWork\\show\\show-%04d.png' % cnt, imgsMerge)
        cnt += 1
        print(cnt)

if __name__ == '__main__':
    #test_video('E:\\TempWork\\c\\c', 'E:\\TempWork\\cq\\cq', 500)
    #merge()
    test_img('E:\\test03.jpg', 'E:\\out03.jpg')