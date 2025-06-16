import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
def load_and_threshold(image_path, threshold=128):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Thresholding
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def calculate_similarity(mask1, mask2):
    # Calculate Hamming distance
    hamming_distance = np.count_nonzero(mask1 != mask2)
    total_pixels = mask1.size
    similarity = (1 - hamming_distance / total_pixels) * 100
    return similarity

def normalize_pil(pre, gt):
    gt = np.asarray(gt)
    pre = np.asarray(pre)
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.5, 1, 0)
    # pre = pre / (pre.max() + 1e-8)
    # pre = np.where(pre > 0.5, 1, 0)

    max_pre = pre.max()
    min_pre = pre.min()
    if max_pre == min_pre:
        pre = pre / 255
    else:
        pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt

# def cal_iou(p1, p2):
#     bp1 = (p1 > 0.5)
#     bp2 = (p2 > 0.5)
#
#     inter = np.sum(bp1 * bp2)
#     union = np.sum(((bp1 + bp2) > 0).astype(np.int8))
#     iou = inter * 1. / union
#     # print(inter, union, iou)
#     return iou


def cal_iou(p1, p2, threshold=128):
    # 将像素值大于 threshold 的区域视为目标（1），其他区域视为背景（0）
    bp1 = (p1 > threshold)
    bp2 = (p2 > threshold)

    # 计算交集（Intersection）
    inter = np.sum(bp1 & bp2)  # 交集是两者都为1的区域
    # 计算并集（Union）
    union = np.sum(bp1 | bp2)  # 并集是至少一个为1的区域
    # 计算IoU
    iou = inter / float(union) if union != 0 else 0  # 避免除以0的情况
    return iou

def blend_images(image1, image2):
    # Simple blending (average)
    blended_image = cv2.addWeighted(image1, 1, image2, 1, 0)
    return blended_image


def image_intersection(image1, image2):
    # 确保两个图像具有相同的尺寸
    if image1.shape != image2.shape:
        raise ValueError("Images do not have the same shape")

     # 创建一个与输入图像相同大小和类型的零矩阵
    height, width = image1.shape[:2]  # 假设是二维或三维图像（彩色或灰度）
    image = np.zeros((height, width), np.uint8)

    # 逐像素计算交集（逻辑与操作），这里我们简单地使用逐元素的最小值
    # 注意：对于图像交集来说，这实际上不是传统意义上的交集，但可以用于某些场景
    # 如果你想要像素值同时非零的位置才设为非零，可以考虑使用逻辑与
    image = np.minimum(image1, image2)

    # 如果你想要更严格的“交集”定义（即两个图像在同一位置都为非零），则可以使用：
    # image = np.where((image1 > 0) & (image2 > 0), 255, 0).astype(np.uint8)
    # 这会将两个图像中非零重叠的像素设为255，其余设为0

    return image

if __name__ == "__main__":


    # 设置计数器
    count = 0
    max_iterations = 5  # 指定的最大循环次数


    maskpath_point = r'./figures/DUTS-TR-opt3-point-vit_l'
    #'./figures/DUTS-TR-opt3-point-96-0-0/'
    maskpath_box =r'./figures/DUTS-TR-opt-box-vit_l/'

    maskpath_sam = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-50-opt3(96)-iou'


    maskpath_A2S_USOD = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-baseA2S-iou50)/final/'


    savepath = r'/data/dataset/DUTS-TR/'

    # /data/dataset/DUTS-TR/DUTS-TR-Mask

    # savepath = r'./figures/DUTS-TR/mask/'


    # 获取文件夹中所有点提示图片文件的路径
    point_mask_image = [os.path.join(maskpath_point, f) for f in os.listdir(maskpath_point) if f.endswith('point_mask.png')]

    # 获取文件夹中所有sam图片文件的路径
    sam_mask_image = [os.path.join(maskpath_sam, f) for f in os.listdir(maskpath_sam)]



    '''第二阶段融合'''
    for point_image_path in point_mask_image:
        print(f"load image: {point_image_path}")
        image_name = point_image_path.split('/')[-1].split('_point_mask.png')[0]
        box_image_path = maskpath_box+image_name+'_box_mask.png'

        # 点提示图
        point_image = cv2.imread(point_image_path, cv2.IMREAD_GRAYSCALE)
        if point_image is None:
            print(f"Failed to load image: {point_image_path}")
            continue

        # box提示
        box_image = cv2.imread(box_image_path, cv2.IMREAD_GRAYSCALE)
        if box_image is None:
            print(f"Failed to load image: {box_image_path}")
            continue

        # point,box = normalize_pil(point_image,box_image)
        # similarity = cal_iou(point,box)

        similarity = cal_iou(point_image, box_image)

        print(f"Similarity between the masks: {similarity:.2f}%")



        if similarity > 0.5:
            '''这里先注释了'''
            blended_image = blend_images(point_image, box_image)
            # blended_image=image_intersection(point_image, box_image)
            # mask_uint8 = blended_image.astype(np.uint8)
            # mask_name = image_name.split('.jpg')[0]+'_sam.jpg'
            if not os.path.exists(savepath + 'DUTS-TR-Mask-vit_l-50'):
                os.makedirs(savepath + 'DUTS-TR-Mask-vit_l-50')

            cv2.imwrite(savepath + 'DUTS-TR-Mask-vit_l-50/'+image_name.split('.jpg')[0]+'.png', blended_image)

            # 图片名称
            image_name = image_name
            # # 掩码名称
            # mask_name = image_name

            list_file = savepath + 'train-DUTS-TR-Mask-vit_l-50.txt'

            # train_dataset = ImageDataTrain(data_root='/path/to/data_root', data_list='/path/to/data_list.txt',
            #                                mask_list_file=mask_list_file)

            # # 假设有一张掩码图片需要保存
            # mask_image = ...  # Your mask image array
            # mask_name = 'mask_image_001.png'

            # # 保存掩码图片，并将路径记录到文本文件中
            # train_dataset.save_mask_image(mask_image, mask_name)

            # with open(list_file, 'a') as f:
            #     f.truncate(0)

            with open(list_file, 'a') as f:
                name = 'DUTS-TR-Image/'+image_name +' '+'DUTS-TR-Mask-vit_l-50/'+image_name+ "\n"
                f.write(name)
            '''这里先注释了'''

            '''
            plt.figure(figsize=(10, 10))
            plt.imshow(blended_image, cmap='gray')
            plt.title(f"similarity: {similarity:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
            '''
            count = count + 1

        else:
            point_image_name = point_image_path.split('/')[-1]
            box_image_name = point_image_path.split('/')[-1].split('_point_mask.png')[0] + '_box_mask.png'
            if not os.path.exists(savepath + 'DUTS-TR-Mask-Deleted-vit_l-50'):
                os.makedirs(savepath + 'DUTS-TR-Mask-Deleted-vit_l-50')
            cv2.imwrite(savepath + 'DUTS-TR-Mask-Deleted-vit_l-50/'+point_image_name,point_image)
            cv2.imwrite(savepath + 'DUTS-TR-Mask-Deleted-vit_l-50/' + box_image_name,box_image)
    print(count)
    '''第二阶段融合'''
        # # 增加计数器
        # count += 1
        # if count % 300 == 299:
        #     print('%d picture' % (count))
        #
        # # 判断是否达到最大循环次数
        # if count >= max_iterations:
        #     break


    '''这是模型筛选相关代码'''
    # for sam_image_path in sam_mask_image:
    #     print(f"load image: {sam_image_path}")
    #     image_name = sam_image_path.split('/')[-1].split('.png')[0]
    #     # image_name=os.path.splitext(os.path.basename(sam_image_path))[0]
    #     A2S_image_path = maskpath_A2S_USOD + image_name + '.png'
    #
    #     # sam提示图
    #     sam_image = cv2.imread(sam_image_path, cv2.IMREAD_GRAYSCALE)
    #     if sam_image is None:
    #         print(f"Failed to load image: {sam_image_path}")
    #         continue
    #
    #     # A2S提示
    #     A2S_image = cv2.imread(A2S_image_path, cv2.IMREAD_GRAYSCALE)
    #     if A2S_image is None:
    #         print(f"Failed to load image: {A2S_image_path}")
    #         continue
    #
    #     similarity = cal_iou(sam_image, A2S_image)
    #
    #     print(f"Similarity between the masks: {similarity:.2f}%")
    #
    #     if similarity > 0.95:
    #         blended_image = blend_images(sam_image, A2S_image)
    #
    #         # mask_uint8 = blended_image.astype(np.uint8)
    #         # mask_name = image_name.split('.jpg')[0]+'_sam.jpg'
    #
    #         Aname = 'DUTS-TR-Mask-Mix-base-SAM-iou-95'
    #         if not os.path.exists(savepath + Aname):
    #             os.makedirs(savepath + Aname)
    #
    #
    #         cv2.imwrite(savepath + Aname+'/' + image_name +'.png', blended_image)
    #
    #         # 图片名称
    #         image_name = image_name
    #         # # 掩码名称
    #         # mask_name = image_name
    #
    #         list_file = savepath + 'train-Mix-base-SAM-iou-95.txt'
    #
    #         # train_dataset = ImageDataTrain(data_root='/path/to/data_root', data_list='/path/to/data_list.txt',
    #         #                                mask_list_file=mask_list_file)
    #
    #         # # 假设有一张掩码图片需要保存
    #         # mask_image = ...  # Your mask image array
    #         # mask_name = 'mask_image_001.png'
    #
    #         # # 保存掩码图片，并将路径记录到文本文件中
    #         # train_dataset.save_mask_image(mask_image, mask_name)
    #
    #         # with open(list_file, 'a') as f:
    #         #     f.truncate(0)
    #
    #         with open(list_file, 'a') as f:
    #             name = 'DUTS-TR-Image/' + image_name + ' '+ Aname+'/' + image_name + "\n"
    #             f.write(name)
    #
    #         '''
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(blended_image, cmap='gray')
    #         plt.title(f"similarity: {similarity:.3f}", fontsize=18)
    #         plt.axis('off')
    #         plt.show()
    #         '''
    #
    #
    #     else:
    #         print("Masks are not similar enough to blend.")

    '''这是模型筛选相关代码'''

        # # 增加计数器
        # count += 1
        # if count % 300 == 299:
        #     print('%d picture' % (count))

        # # 判断是否达到最大循环次数
        # if count >= max_iterations:
        #     break