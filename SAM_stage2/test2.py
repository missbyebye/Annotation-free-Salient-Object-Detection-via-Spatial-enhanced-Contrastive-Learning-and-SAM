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

# def calculate_similarity(mask1, mask2):
#     # Calculate Hamming distance
#     hamming_distance = np.count_nonzero(mask1 != mask2)
#     total_pixels = mask1.size
#     similarity = (1 - hamming_distance / total_pixels) * 100
#     return similarity
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
    blended_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    return blended_image
if __name__ == "__main__":

    maskpath_point = r'./figures/DUTS-TR-opt3-point-96-0-0/'
    maskpath_box =r'./figures/DUTS-TR-opt3-box-96/'

    maskpath_sam = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-50-opt3(96)-iou/'
    maskpath_A2S_USOD = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-baseA2S-iou50)/final/'


    savepath = r'/data/segment-anything-main/segment-anything-main/figures/test'

    figpath = r'./figures/test-image/'

    # 获取文件夹中所有原始图片文件的路径
    raw_image_files = [os.path.join(figpath, f) for f in os.listdir(figpath)]
    raw_image_files = ['./figures/test-image/ILSVRC2012_test_00004908.jpg']

    # 获取文件夹中所有点提示图片文件的路径
    point_mask_image = [os.path.join(maskpath_point, f) for f in os.listdir(maskpath_point) if f.endswith('point_mask.jpg')]

    # # 获取文件夹中所有sam图片文件的路径
    # sam_mask_image = [os.path.join(maskpath_sam, f) for f in os.listdir(maskpath_sam)]


    for raw_image in raw_image_files:
        print(f"load image: {raw_image}")
        raw_image = raw_image.split('/')[-1]
        box_image_path =maskpath_box+raw_image+'_box_mask.jpg'
        point_image_path = maskpath_point+raw_image + '_point_mask.jpg'

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

        print(f"Similarity between the masks: {similarity:.3f}%")

        blended_image = blend_images(point_image, box_image)

        plt.figure(figsize=(10, 10))
        plt.imshow(blended_image, cmap='gray')
        # plt.title(f"similarity: {similarity:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(savepath+'/'+raw_image+'-iou.png', bbox_inches='tight', pad_inches=0)
        plt.show()
        cv2.imwrite(savepath +'/'+ raw_image +'point.png' , point_image)
        cv2.imwrite(savepath +'/'+ raw_image + 'box.png', box_image)


    '''  '''
    for raw_image in raw_image_files:
        print(f"load image: {raw_image}")
        raw_image = raw_image.split('/')[-1].split('.jpg')[0]
        # image_name=os.path.splitext(os.path.basename(sam_image_path))[0]
        A2S_image_path = maskpath_A2S_USOD + raw_image + '.png'
        sam_image_path = maskpath_sam + raw_image + '.png'
        # sam提示图
        sam_image = cv2.imread(sam_image_path, cv2.IMREAD_GRAYSCALE)
        if sam_image is None:
            print(f"Failed to load image: {sam_image_path}")
            continue

        # A2S提示
        A2S_image = cv2.imread(A2S_image_path, cv2.IMREAD_GRAYSCALE)
        if A2S_image is None:
            print(f"Failed to load image: {A2S_image_path}")
            continue

        similarity = cal_iou(sam_image, A2S_image)

        print(f"Similarity between the masks: {similarity:.3f}%")
        blended_image = blend_images(sam_image, A2S_image)
        plt.figure(figsize=(10, 10))
        plt.imshow(blended_image, cmap='gray')
        plt.axis('off')
        plt.savefig(savepath+'/'+raw_image+'-base_sam_iou.png', bbox_inches='tight', pad_inches=0)
        plt.show()
        cv2.imwrite(savepath + '/' + raw_image + 'sam.png', sam_image)
        cv2.imwrite(savepath + '/' + raw_image + 'A2S.png', A2S_image)
    #
    # maskpath_deleted = r'./figures/test-image/'
    # maskpath_A2S_USOD = r'/data/dataset/DUTS-TR/DUTS-TR-Mask-baseA2S-iou50)/final/'
    #
    # '''     '''
    # # 获取文件夹中所有deleted图片文件的路径
    # deleted_mask_list_point = [os.path.join(maskpath_deleted, f) for f in os.listdir(maskpath_deleted) if
    #                            f.endswith('point_mask.png')]
    #
    # for deleted_mask_path_point in deleted_mask_list_point:
    #     print(f"load image: {deleted_mask_path_point}")
    #     image_name = deleted_mask_path_point.split('/')[-1].split('.jpg')[0]
    #     A2S_image_path = maskpath_A2S_USOD + image_name + '.png'
    #
    #     # deleted提示图
    #     deleted_mask_point = cv2.imread(deleted_mask_path_point, cv2.IMREAD_GRAYSCALE)
    #     deleted_mask_path_box = deleted_mask_path_point.split('_point_mask.png')[0] + '_box_mask.png'
    #     deleted_mask_box = cv2.imread(deleted_mask_path_box, cv2.IMREAD_GRAYSCALE)
    #
    #     if deleted_mask_point is None:
    #         print(f"Failed to load image: {deleted_mask_path_point}")
    #         continue
    #     if deleted_mask_box is None:
    #         print(f"Failed to load image: {deleted_mask_path_box}")
    #         continue
    #
    #     # A2S提示
    #     A2S_image = cv2.imread(A2S_image_path, cv2.IMREAD_GRAYSCALE)
    #     if A2S_image is None:
    #         print(f"Failed to load image: {A2S_image_path}")
    #         continue
    #
    #     similarity_point_A2S = cal_iou(deleted_mask_point, A2S_image)
    #     similarity_box_A2S = cal_iou(deleted_mask_box, A2S_image)
    #
    #     print(f"similarity_point_A2S: {similarity_point_A2S:.3f}%")
    #     print(f"similarity_box_A2S: {similarity_box_A2S:.3f}%")
    #
    #     blended_image = blend_images(deleted_mask_box, A2S_image)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(blended_image, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(savepath + '/' + image_name + 'box-base_delete_iou.png', bbox_inches='tight', pad_inches=0)
    #     plt.show()
    #
    #     blended_image = blend_images(deleted_mask_point, A2S_image)
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(blended_image, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(savepath + '/' + image_name + 'point-base_delete_iou.png', bbox_inches='tight', pad_inches=0)
    #     plt.show()



