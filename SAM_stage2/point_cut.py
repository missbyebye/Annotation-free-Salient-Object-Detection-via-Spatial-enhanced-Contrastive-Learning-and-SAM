import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import random
import sys
import math
# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
# CUDA_LAUNCH_BLOCKING=1.
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = "cuda:0"

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



figpath = r'/data/ccam_master/data/DUTS-TR/DUTS-TR/DUTS-TR-Image'
maskpath = r'/data/CCAM/WSSS_Saliency/vis_cam3/CCAM_DUTS_MOCO@train@scale=0.5,1.0,1.5,2.0/'

# save_figpath= r'/data/dataset/DUTS-TR-point-cut'
save_figpath1= r'./figures/DUTS-TR-opt3-point-vit_l/'
if not os.path.exists(save_figpath1):
    os.makedirs(save_figpath1)

# 获取文件夹中所有原始图片文件的路径
raw_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath)]
# raw_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath) if f.endswith('raw.jpg')]
# cam_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath) if f.endswith('cam.jpg')]


# # 找到文件的位置
# try:
#     index = raw_image_files.index(r'/data/ccam_master/data/DUTS-TR/DUTS-TR/DUTS-TR-Image/n07720875_3279.jpg_point_mask.jpg')
# except ValueError:
#     print("图片不存在")
#     index = -1
#
# # 如果照片存在，删除它之前的所有元素
# if index != -1:
#     raw_image_files = raw_image_files[index:]

# 设置计数器
count = 0
max_iterations = 3  # 指定的最大循环次数

for raw_image in raw_image_files:
    print(f"load image: {raw_image}")

    # # 使用 split 取名字
    # image_name = raw_image.split('_raw')[0]
    image_name = raw_image.split('/')[-1]
    # print(image_name)

    cam_image = maskpath + image_name+'_map.png'

    # # 没有掩码则跳过
    # if len(cam_image) > 1:
    #     filename_before_raw = cam_image[0]
    # else:
    #     continue

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


    # # 两个点作提示
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask1)
    # print(max_val,max_loc,min_val,min_loc)

    # # 6个点作提示
    # mask_point = mask1.copy()
    # # 找到最大的3个点
    # max_locations = []
    # for _ in range(3):
    #     _, max_val, _, max_loc = cv2.minMaxLoc(mask_point)
    #     print(max_val)
    #     max_locations.append(max_loc)
    #     mask_point[max_loc[1], max_loc[0]] = 125  # 将找到的点置零，以便找到下一个最大值
    #
    # # 找到最小的3个点
    # min_locations = []
    # for _ in range(3):
    #     min_val, _, min_loc,_ = cv2.minMaxLoc(mask_point, None)
    #     min_locations.append(min_loc)
    #     mask_point[min_loc[1], min_loc[0]] = 125  # 将找到的点置为最大值，以便找到下一个最小值
    #
    # # 将最大和最小的点合并
    # input_point = np.array(max_locations + min_locations)
    # print("最大的3个点：", max_locations)
    # print("最小的3个点：", min_locations)

    input_point = []
    '''
    # 随机点+最大最小点+作提示
    # 生成随机点的数量

    num_points = 16

    # 随机选取点加最大最小点
    for _ in range(num_points):
        # 随机生成行和列索引
        rand_row = random.randint(0, height-1)
        rand_col = random.randint(0, width-1)
        input_point.append((rand_col, rand_row))  # 注意OpenCV中坐标是 (列, 行)

    # print("随机选取的四个点：", input_point)
    '''
    # 提示点-窗格
    num_points_1 = 16
    mask_height, mask_width = image2.shape[:2]

    num = int(math.sqrt(num_points_1))
    for i in range(num):
        for j in range(num):
            # # 生成左上角点的位置
            # rand_start_row = int(mask_height/num*i)
            # rand_start_col = int(mask_width/num*j)
            # 
            # # 在 6x6 的区域内随机选择一个点
            # rand_row = random.randint(rand_start_row, rand_start_row + int(mask_height/num))
            # rand_col = random.randint(rand_start_col, rand_start_col + int(mask_width/num))

            rand_row = int(mask_height/num*(i+0.5))
            rand_col = int(mask_width/num*(j+0.5))

            input_point.append((rand_col, rand_row))  # 注意OpenCV中坐标是 (列, 行)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask1)
    min_max_point=[min_loc,max_loc]
    input_point=np.array(input_point+min_max_point)


    # 掩码图转换为01数值
    maskChange = mask1.copy()
    plt.imshow(mask1)
    mask_input = np.array(maskChange)


    mask_condition_front = mask_input >= 96
    mask_condition_back = mask_input <=0

    # 前景
    maskChange_eroded = mask1.copy()
    maskChange_eroded[:]=0
    maskChange_eroded[mask_condition_front]=1

    # 背景
    maskChange_dilated = mask1.copy()
    maskChange_dilated[:]=0
    maskChange_dilated[mask_condition_back]=1


    # # 二值图腐蚀-前景范围
    # kernel = np.ones((10, 10), np.uint8)  # 定义一个 5x5 的核，可以根据实际情况调整大小
    # maskChange_eroded = cv2.erode(maskChange, kernel, iterations=1)
    # # plt.figure(figsize=(10,10))
    # # plt.imshow(maskChange_eroded,cmap='gray')
    # # plt.show()
    #
    # # 二值图膨胀反转-背景范围
    # maskChange_dilated = cv2.dilate(maskChange, kernel, iterations=1)
    # maskChange_dilated = np.logical_not(maskChange_dilated).astype(np.uint8)
    # # plt.figure(figsize=(10,10))
    # # plt.imshow(maskChange_dilated, cmap='gray')
    # # plt.show()


    # #提示点-最大最小点
    # input_point = np.array([max_loc,min_loc])
    # input_label = np.array([maskChange[max_loc[1], max_loc[0]], maskChange[min_loc[1], min_loc[0]]])


    # 提示点
    # 提取对应的 label
    input_label = []
    # 被移除的点的坐标
    points_to_remove = []
    # 被移除的点的索引
    points_to_remove_index=[]

    # for loc in input_point:
    #
    #     input_label.append(maskChange[loc[1], loc[0]])
    #
    # input_label = np.array(input_label)
    #


    for i,loc in enumerate(input_point):
        x, y = loc
        # print(loc)
        if maskChange_eroded[y, x] == 1:
            input_label.append(1)
        elif maskChange_dilated[y, x] == 1:
            input_label.append(0)
        else:
            points_to_remove.append(loc)
            points_to_remove_index.append(i)

    # 将 input_label 转换为 numpy 数组
    input_label = np.array(input_label)
    # 将被移除的点的索引从大到小排列
    points_to_remove_index=np.sort(points_to_remove_index)[::-1]
    # print(points_to_remove)
    # 从 input_point 中删除不在 maskChange_eroded 和 maskChange_dilated 范围内的点
    for i in points_to_remove_index:
        input_point=np.delete(input_point,i, axis=0)
    # print(input_point,input_label)
    # print(len(input_point),len(input_label))

    '''绘图部分'''

    # #原图和点
    # plt.figure(figsize=(10,10))
    # plt.imshow(maskChange_eroded, cmap='gray')
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    # #原图和点
    # plt.figure(figsize=(10,10))
    # plt.imshow(maskChange_dilated, cmap='gray')
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()
    #
    # #原图和点
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()
    '''绘图部分'''

    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    max_score_index = np.argmax(scores)


    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     '''绘图部分'''
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     if i==max_score_index:
    #         print(i,'it')
    #
    #
    #
    #     '''绘图部分'''
    #     plt.show()
    #
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(mask, cmap='gray')
    #     plt.show()
    # print(max_score_index)

    ''' '''
    #保存
    mask_uint8 = (masks[max_score_index] * 255).astype(np.uint8)
    cv2.imwrite(save_figpath1 + image_name + '_point_mask.png', mask_uint8)
    print(count)
    if os.path.exists(save_figpath1 + image_name + '_point_mask.png'):

        print("图片保存成功！")
    else:
        print("图片保存失败！")



    # 增加计数器
    count += 1
    if count  % 300 == 299:
        print('%d picture' % (count))

    # # # 判断是否达到最大循环次数
    # if count >= max_iterations:
    #     break
