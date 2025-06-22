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

def blend_images(image1, image2):
    # Simple blending (average)
    blended_image = cv2.addWeighted(image1, 1, image2, 1, 0)
    return blended_image

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



if __name__ == "__main__":

    # 设置计数器
    count = 0
    # max_iterations = 100  # 指定的最大循环次数

    maskpath_deleted = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-Deleted-iou50)/'
    maskpath_A2S_USOD = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-baseA2S-iou50)/final/'

    savepath = r'/data/dataset/DUTS-TR/'

    # 获取文件夹中所有deleted图片文件的路径
    deleted_mask_list_point = [os.path.join(maskpath_deleted, f) for f in os.listdir(maskpath_deleted) if f.endswith('point_mask.png')]



    for deleted_mask_path_point in deleted_mask_list_point:
        print(f"load image: {deleted_mask_path_point}")
        image_name = deleted_mask_path_point.split('/')[-1].split('.jpg')[0]
        A2S_image_path = maskpath_A2S_USOD + image_name + '.png'

        # deleted提示图
        deleted_mask_point = cv2.imread(deleted_mask_path_point, cv2.IMREAD_GRAYSCALE)
        deleted_mask_path_box = deleted_mask_path_point.split('_point_mask.png')[0]+'_box_mask.png'
        deleted_mask_box = cv2.imread(deleted_mask_path_box, cv2.IMREAD_GRAYSCALE)

        if deleted_mask_point is None:
            print(f"Failed to load image: {deleted_mask_path_point}")
            continue
        if deleted_mask_box is None:
            print(f"Failed to load image: {deleted_mask_path_box}")
            continue

        # A2S提示
        A2S_image = cv2.imread(A2S_image_path, cv2.IMREAD_GRAYSCALE)
        if A2S_image is None:
            print(f"Failed to load image: {A2S_image_path}")
            continue
        # else:
        #     # 二值化图像，阈值设为127，最大值为255
        #     # cv2.threshold() 返回两个值，第二个值为二值化后的图像
        #     _, A2S_image = cv2.threshold(A2S_image, 128, 255, cv2.THRESH_BINARY)

        similarity_point_A2S = cal_iou(deleted_mask_point, A2S_image)
        similarity_box_A2S = cal_iou(deleted_mask_box, A2S_image)

        print(f"similarity_point_A2S: {similarity_point_A2S:.2f}%")
        print(f"similarity_box_A2S: {similarity_box_A2S:.2f}%")

        if max(similarity_point_A2S,similarity_box_A2S) > 0.95:
            if similarity_point_A2S > similarity_box_A2S:
                image_to_blend = deleted_mask_point
            else:
                image_to_blend = deleted_mask_box
#修改---------------------
            # blended_image = blend_images(image_to_blend, A2S_image)
            blended_image = image_to_blend

            if not os.path.exists(savepath + 'DUTS-TR-Mask-Mix-base-Deleted-iou95'):
                os.makedirs(savepath + 'DUTS-TR-Mask-Mix-base-Deleted-iou95')
            cv2.imwrite(savepath + 'DUTS-TR-Mask-Mix-base-Deleted-iou95/' + image_name +'.png', blended_image)

            # 图片名称
            image_name = image_name

            list_file = savepath + 'train-Mix-2base-Deleted-iou95.txt'


            with open(list_file, 'a') as f:
                name = 'DUTS-TR-Image/' + image_name + '.jpg ' + 'DUTS-TR-Mask-Mix-base-Deleted-iou95/' + image_name + ".png\n"
                f.write(name)


        else:
            print("Masks are not similar enough to blend.")


        # 增加计数器
        count += 1
        if count % 300 == 299:
            print('%d picture' % (count))

        # # 判断是否达到最大循环次数
        # if count >= max_iterations:
        #     break