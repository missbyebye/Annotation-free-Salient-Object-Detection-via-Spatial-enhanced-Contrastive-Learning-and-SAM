import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import random
import sys
import tqdm


# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = "cuda:2"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    # 获取矩形框的左上角坐标和宽度、高度
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # 在坐标轴上添加矩形框
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0 ,0 ,0 ,0), lw=2))

figpath = r'/data/ccam_master/data/DUTS-TR/DUTS-TR/DUTS-TR-Image'
maskpath = r'/data/CCAM/WSSS_Saliency/vis_cam3/CCAM_DUTS_MOCO@train@scale=0.5,1.0,1.5,2.0/'
# save_figpath= r'/data/dataset/DUTS-TR-point-cut'
save_figpath1= r'./figures/DUTS-TR-opt-box-vit_l/'
if not os.path.exists(save_figpath1):
    os.makedirs(save_figpath1)

# 获取文件夹中所有原始图片文件的路径
raw_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath) ]
# cam_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath) if f.endswith('cam.jpg')]

# # 找到文件的位置
# try:
#     index = raw_image_files.index(r'/data/ccam_master/data/DUTS-TR/DUTS-TR/DUTS-TR-Image/n04542943_2056.jpg')
# except ValueError:
#     print("图片不存在")
#     index = -1
#
# # 如果照片存在，删除它之前的所有元素
# if index != -1:
#     raw_image_files = raw_image_files[index:]

# 设置计数器
count = 0
max_iterations = 10  # 指定的最大循环次数

for raw_image in raw_image_files:
    print(f"load image: {raw_image}")

    # 图片名字
    image_name = raw_image.split('/')[-1]
    # print(image_name)
    # 掩码路径
    cam_image = maskpath + image_name+'_map.png'

    # 原图
    image = cv2.imread(raw_image, cv2.COLOR_BGR2RGB)
    if image is None:
        print(f"Failed to load image: {raw_image}")
        continue


    #掩码
    mask1 = cv2.imread(cam_image,cv2.IMREAD_GRAYSCALE)
    if mask1 is None:
        print(f"Failed to load image: {cam_image}")
        continue


    # 原图灰度图
    image2 = cv2.imread(raw_image, cv2.IMREAD_GRAYSCALE)
    height, width = image2.shape[:2]
    new_shape = (width, height)
    # 调整掩码图片到原图大小
    mask1 = cv2.resize(mask1, new_shape)


    # 掩码图转换为01数值
    maskChange = mask1.copy()
    plt.imshow(mask1)
    mask_input = np.array(maskChange)
    mask_condition = mask_input > 128
    maskChange[:]=0
    maskChange[mask_condition]=1

    # input_box = np.array([50, 37, 170, 200])

    x, y, w, h = cv2.boundingRect(maskChange)
    # contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    input_box=np.array([x,y,x+w,y+h])


    ''' 
    #二值化图和方框
    plt.figure(figsize=(10,10))
    plt.imshow(maskChange, cmap='gray')
    show_box(input_box, plt.gca())
    plt.axis('on')
    plt.show()

    #原图和方框
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_box(input_box, plt.gca())
    plt.axis('on')
    plt.show()
    '''
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):

        ''' 
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_box(input_box, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        '''
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(save_figpath1 + image_name + '_box_mask.png', mask_uint8)
        print("save",save_figpath1 + image_name + '_box_mask.png')
        ''' 
        plt.show()

        # 掩码图片
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='gray')
        plt.show()
        '''
        # print(count)
        # if os.path.exists(image_name + 'point_mask.jpg'):
        #     print("图片保存成功！")
        # else:
        #     print("图片保存失败！")


    # 增加计数器
    count += 1
    if count  % 300 == 299:
        print('%d picture' % (count))

    # # 判断是否达到最大循环次数
    # if count >= max_iterations:
    #     break