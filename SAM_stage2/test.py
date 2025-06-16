
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import random
import sys
import math


sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

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


def show_points(coords, labels, ax, flag,marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if flag=="front":
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white',
               linewidth=1.25)
    elif flag=='back':
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_points1(coords,  ax, marker_size=375):
    points = coords
    ax.scatter(points[:, 0], points[:, 1], color='yellow', marker='o', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    # 获取矩形框的左上角坐标和宽度、高度
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # 在坐标轴上添加矩形框
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0 ,0 ,0 ,0), lw=6))



figpath = r'./figures/test-image/'
maskpath = r'/data/CCAM/WSSS_Saliency/vis_cam3/CCAM_DUTS_MOCO@train@scale=0.5,1.0,1.5,2.0/'

# save_figpath= r'/data/dataset/DUTS-TR-point-cut'
save_figpath1= r'./figures/test/'

# 获取文件夹中所有原始图片文件的路径
raw_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath)]
raw_image_files=['./figures/test-image/ILSVRC2012_test_00000172.jpg']


for raw_image in raw_image_files:
    print(f"load image: {raw_image}")

    image_name = raw_image.split('/')[-1]
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


    input_point = []
    # 提示点-窗格
    num_points_1 = 16
    mask_height, mask_width = image2.shape[:2]

    num = int(math.sqrt(num_points_1))
    for i in range(num):
        for j in range(num):
            rand_row = int(mask_height/num*(i+0.5))
            rand_col = int(mask_width/num*(j+0.5))

            input_point.append((rand_col, rand_row))  # 注意OpenCV中坐标是 (列, 行)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask1)
    min_max_point=[min_loc,max_loc]
    input_point=np.array(input_point+min_max_point)


    # 掩码图转换为01数值
    maskChange = mask1.copy()
    # plt.imshow(mask1)
    mask_input = np.array(maskChange)
    mask_condition = mask_input > 128
    maskChange[:]=0
    maskChange[mask_condition]=1

    mask_condition_front = mask_input > 160
    mask_condition_back = mask_input < 96

    # 前景
    maskChange_eroded = mask1.copy()
    maskChange_eroded[:]=0
    maskChange_eroded[mask_condition_front]=1
    # 背景
    maskChange_dilated = mask1.copy()
    maskChange_dilated[:]=1
    maskChange_dilated[mask_condition_back]=0
    #全部
    maskChange_all = (maskChange_dilated+maskChange_eroded)/2


    # 提示点
    # 提取对应的 label
    input_label = []
    # 被移除的点的坐标
    points_to_remove = []
    # 被移除的点的索引
    points_to_remove_index=[]
    points_to_remove_label=[]

    for i,loc in enumerate(input_point):
        x, y = loc

        if maskChange_eroded[y, x] == 1:
            input_label.append(1)
        elif maskChange_dilated[y, x] == 1:
            input_label.append(0)
        else:
            points_to_remove.append(loc)
            points_to_remove_label.append(input_point[i])
            points_to_remove_index.append(i)

    # 将 input_label 转换为 numpy 数组
    input_label = np.array(input_label)
    # 将被移除的点的索引从大到小排列
    points_to_remove_index=np.sort(points_to_remove_index)[::-1]
    # print(points_to_remove)
    # 从 input_point 中删除不在 maskChange_eroded 和 maskChange_dilated 范围内的点
    input_point1=input_point
    for i in points_to_remove_index:
        input_point1=np.delete(input_point1,i, axis=0)

    # points_to_remove = np.array(points_to_remove+input_point)



    '''绘图部分'''
    #前景
    plt.figure(figsize=(10,10))
    plt.imshow(mask1, cmap='gray')
    show_points1(input_point, plt.gca())
    print(input_point1.shape)
    print(input_label.shape)
    show_points(input_point1, input_label, plt.gca(), "back")
    show_points(input_point1, input_label, plt.gca(), "front")
    # 移除坐标轴
    plt.axis('off')
    # 保存图片
    plt.savefig(save_figpath1+image_name+'-map.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    # #前景
    # plt.figure(figsize=(10,10))
    # plt.imshow(maskChange_eroded, cmap='gray')
    # show_points1(input_point,  plt.gca())
    # show_points(input_point1, input_label, plt.gca(),"front")
    # plt.axis('off')
    # # 保存图片
    # plt.savefig(save_figpath1+image_name+'-front.png', bbox_inches='tight', pad_inches=0)
    # # plt.axis('on')
    # plt.show()
    #
    # #背景
    # plt.figure(figsize=(10,10))
    # plt.imshow(maskChange_dilated, cmap='gray')
    # # show_points1(input_point,  plt.gca())
    # show_points(input_point1, input_label, plt.gca(),"back")
    # plt.axis('off')
    # # 保存图片
    # plt.savefig(save_figpath1+image_name+'-back.png', bbox_inches='tight', pad_inches=0)
    # # plt.axis('on')
    # plt.show()
    #全景
    plt.figure(figsize=(10,10))
    plt.imshow(maskChange_all, cmap='gray')
    show_points1(input_point,  plt.gca())
    show_points(input_point1, input_label, plt.gca(),"front")
    show_points(input_point1, input_label, plt.gca(), "back")
    plt.axis('off')
    # 保存图片
    plt.savefig(save_figpath1+image_name+'-all.png', bbox_inches='tight', pad_inches=0)
    # plt.axis('on')
    plt.show()

    #图片
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points1(input_point, plt.gca())
    show_points(input_point1, input_label, plt.gca(), "back")
    show_points(input_point1, input_label, plt.gca(), "front")

    # plt.axis('on')
    plt.show()


    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    max_score_index = np.argmax(scores)
'''
    mask_uint8 = (masks[max_score_index] * 255).astype(np.uint8)
    cv2.imwrite(save_figpath1 + image_name + '_point_mask.jpg', mask_uint8)

    print(count)
    if os.path.exists(save_figpath1 + image_name + '_point_mask.jpg'):
        print("图片保存成功！")
    else:
        print("图片保存失败！")
'''
# for raw_image in raw_image_files:
#     print(f"load image: {raw_image}")
#
#     # 图片名字
#     image_name = raw_image.split('/')[-1]
#     # print(image_name)
#     # 掩码路径
#     cam_image = maskpath + image_name+'_map.png'
#
#     # 原图
#     image = cv2.imread(raw_image, cv2.COLOR_BGR2RGB)
#     if image is None:
#         print(f"Failed to load image: {raw_image}")
#         continue
#
#
#     #掩码
#     mask1 = cv2.imread(cam_image,cv2.IMREAD_GRAYSCALE)
#     if mask1 is None:
#         print(f"Failed to load image: {cam_image}")
#         continue
#
#
#     # 原图灰度图
#     image2 = cv2.imread(raw_image, cv2.IMREAD_GRAYSCALE)
#     height, width = image2.shape[:2]
#     new_shape = (width, height)
#     # 调整掩码图片到原图大小
#     mask1 = cv2.resize(mask1, new_shape)
#
#
#     # 掩码图转换为01数值
#     maskChange = mask1.copy()
#     plt.imshow(mask1)
#     mask_input = np.array(maskChange)
#     mask_condition = mask_input > 128
#     maskChange[:]=0
#     maskChange[mask_condition]=1
#
#     # input_box = np.array([50, 37, 170, 200])
#
#     x, y, w, h = cv2.boundingRect(maskChange)
#     # contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     input_box=np.array([x,y,x+w,y+h])
#
#
#
#     #二值化图和方框
#     plt.figure(figsize=(10,10))
#     plt.imshow(maskChange, cmap='gray')
#     show_box(input_box, plt.gca())
#     plt.axis('off')
#     plt.savefig(save_figpath1 + image_name+'-map-box.png', bbox_inches='tight', pad_inches=0)
#     plt.show()
#
#     #原图和方框
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_box(input_box, plt.gca())
#     plt.axis('off')
#     plt.savefig(save_figpath1 + image_name+'-box.png', bbox_inches='tight', pad_inches=0)
#     plt.show()
#
#     predictor.set_image(image)
#
#     masks, scores, _ = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=input_box[None, :],
#         multimask_output=False,
#     )
#
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#
#         '''
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca())
#         show_box(input_box, plt.gca())
#         plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         '''
#         mask_uint8 = (mask * 255).astype(np.uint8)
#         cv2.imwrite(save_figpath1 + image_name + '_box_mask.jpg', mask_uint8)
#         '''
#         plt.show()
#
#         # 掩码图片
#         plt.figure(figsize=(10, 10))
#         plt.imshow(mask, cmap='gray')
#         plt.show()
#         '''


