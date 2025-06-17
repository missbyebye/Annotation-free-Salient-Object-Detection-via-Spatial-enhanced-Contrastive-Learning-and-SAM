import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import shutil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#mean = np.array((104.00699, 116.66877, 122.67892)).reshape((1, 1, 3))
mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def merge_folders(src_folder1, src_folder2, dest_folder):
    num = 0
    """
    将两个源文件夹的内容合并到一个目标文件夹中。

    参数:
    - src_folder1: 第一个源文件夹路径。
    - src_folder2: 第二个源文件夹路径。
    - dest_folder: 目标文件夹路径。
    """
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 复制第一个源文件夹中的所有文件到目标文件夹
    for item in os.listdir(src_folder1):
        src_path = os.path.join(src_folder1, item)
        dest_path = os.path.join(dest_folder, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            num=num+1

    # 复制第二个源文件夹中的所有文件到目标文件夹
    for item in os.listdir(src_folder2):
        src_path = os.path.join(src_folder2, item)
        dest_path = os.path.join(dest_folder, item)
        # if num>=7900:
        #     break
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            num = num + 1




def get_image_list(name, config, phase):
    images = []
    gts = []
    '''todo 改路径'''
    # 如果是train
    if phase == 'train':
        # image_root = '/data/dataset/DUTS-TR/DUTS-TR-Image'
        # gt_root = '/data/dataset/DUTS-TR/DUTS-TR-Mask-Mix-87-16-A2S-85-1'
        # gts = sorted(
        #     [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])
        # # for gt in gts:
        # #     # 从路径中获取文件名
        # #     image_name = os.path.basename(gt)
        # #     # 去掉文件扩展名
        # #     base_name, _ = os.path.splitext(image_name)
        # #     print('train',base_name)
        # #
        # #     images_image = os.path.join(image_root, base_name)+'.jpg'
        # #     images.append(images_image)
        # # 遍历 GT 路径列表
        # i = 0
        image_root = '/data/dataset/DUTS-TR/DUTS-TR-Image'
        gt_root1 = '/data/dataset/DUTS-TR/DUTS-TR-Mask-Mix-base-SAM-90'
        gt_root2 = '/data/dataset/DUTS-TR/DUTS-TR-Mask-Mix-base-Deleted-80'
        gt_root = '/data/segment-anything-main/segment-anything-main/figures/DUTS-TR-16'
        # merge_folders(gt_root1, gt_root2, gt_root)

        # gts1 = sorted(
        #     [os.path.join(gt_root1, f) for f in os.listdir(gt_root1) if f.endswith('.jpg') or f.endswith('.png')])
        # gts2 = sorted(
        #     [os.path.join(gt_root2, f) for f in os.listdir(gt_root2) if f.endswith('.jpg') or f.endswith('.png')])
        # gts = gts1 + gts2
        gts= sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])


        # 遍历 GT 路径列表
        i = 0
        while i < len(gts):
            gt = gts[i]
            # 从路径中获取文件名
            image_name = os.path.basename(gt)
            # 去掉文件扩展名
            base_name, _ = os.path.splitext(image_name)
            base_name=base_name.split('.jpg')[0]
            # 生成对应的图像路径
            images_image = os.path.join(image_root, base_name) + '.jpg'
            # 检查图像文件是否存在
            if os.path.exists(images_image):
                images.append(images_image)
                i += 1
            else:
                # 如果图像文件不存在，从 gts 列表中删除对应项
                print(f"Removing missing image: {images_image}")
                del gts[i]  # 删除当前项

    else:
        '''这是测试集部分'''
        image_root = '/data/dataset/dataset_test/DUTS-TE/DUTS-TE-Image'
        gt_root = '/data/dataset/dataset_test/DUTS-TE/DUTS-TE-Mask'
        # image_root = '/data/dataset/DUTS-TR/DUTS-TR-Image'
        # gt_root = '/data/dataset/DUTS-TR/DUTS-TR-Mask-87-16'
        gts = sorted(
            [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')])

        # for gt in gts:
        #     # 从路径中获取文件名
        #     image_name = os.path.basename(gt)
        #     # 去掉文件扩展名
        #     base_name, _ = os.path.splitext(image_name)
        #     print('test', base_name)
        #
        #     images_image = os.path.join(image_root, base_name)+'.jpg'
        #     images.append(images_image)

        # 遍历 GT 路径列表
        i = 0
        while i < len(gts):
            gt = gts[i]
            # 从路径中获取文件名
            image_name = os.path.basename(gt)
            # 去掉文件扩展名
            base_name, _ = os.path.splitext(image_name)
            # 生成对应的图像路径
            images_image = os.path.join(image_root, base_name) + '.jpg'
            # 检查图像文件是否存在
            if os.path.exists(images_image):
                images.append(images_image)
                i += 1
            else:
                # 如果图像文件不存在，从 gts 列表中删除对应项
                print(f"Removing missing image: {images_image}")
                del gts[i]  # 删除当前项

        '''这是测试集部分'''

        '''这是预测训练集部分'''
        # # image_root = os.path.join(config['data_path'], name, 'images')
        # image_root = '/data/dataset/DUTS-TR/DUTS-TR-Image'
        # if phase == 'train' and name == 'DUTS-TR':
        #     tag = 'moco'
        # else:
        #     tag = 'segmentations'
        #
        # print(tag)
        # # gt_root = os.path.join(config['data_path'], name, tag)
        # gt_root = '/data/dataset/DUTS-TR/DUTS-TR-Mask-87-16'
        #
        # gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg')])
        # images = []
        # for gt in gts:
        #     # print(f"load image: {gt}")
        #
        #     # # 使用 split 取名字
        #     # image_name = raw_image.split('_raw')[0]
        #     image_name = gt.split('/')[-1]
        #
        #     images_image = image_root + '/' + image_name
        #     images.append(images_image)
        #
        # # images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
        # image_root = os.path.join(config['data_path'], name, 'images')

        # image_root = '/data/dataset/DUTS-TR/DUTS-TR-Image'
        # if phase == 'train' and name == 'DUTS-TR':
        #     tag = 'moco'
        # else:
        #     tag = 'segmentations'
        #
        # print(tag)
        # gt_root = '/data/dataset/DUTS-TR/DUTS-TR-Image'
        #
        # gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg')])
        # images = []
        # for gt in gts:
        #
        #     image_name = gt.split('/')[-1]
        #
        #     image = image_root + '/' + image_name
        #     images.append(image)

        '''这是预测训练集部分'''

        print(images[:5])
        print(gts[:5])
    return images, gts

def get_loader(config):
    dataset = Train_Dataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)


class Train_Dataset(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['trset']
        #if config['stage'] == 1:
        #    self.images, self.gts = get_image_list(config['trset'], config, 'train')
        #    self.size = len(self.images)
        
        self.images_list, self.gts_list = get_image_list(config['trset'], config, 'train')
        self.size = len(self.images_list)
        
        if config['stage'] == 2:
            self.images, self.gts = self.load_data()
            print(len(self.images), len(self.gts))

    def __getitem__(self, index):
        image = Image.open(self.images_list[index]).convert('RGB')
        gt = Image.open(self.gts_list[index]).convert('L')
        
        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size))
    
        image = np.array(image).astype(np.float32)
        gt = np.array(gt)
        
        if random.random() > 0.5:
            image = image[:, ::-1]
            gt = gt[:, ::-1]
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = np.expand_dims(gt / 255., axis=0)
        #gt = np.expand_dims((gt > 128).astype(np.float32), axis=0)
        
        return image, gt
        
    def load_data(self):
        images = []
        gts = []
        for idx in range(self.size):
            image, gt = self.__getitem__(idx)
            images.append(image)
            gts.append(gt)
            
        '''
        for image_path, gt_path in zip(self.images_list, self.gts_list):
            image = Image.open(image_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            
            img_size = self.config['size']
            image = image.resize((img_size, img_size))
            gt = gt.resize((img_size, img_size))
        
            image = np.array(image).astype(np.float32)
            gt = np.array(gt)
            
            if random.random() > 0.5:
                image = image[:, ::-1]
                gt = gt[:, ::-1]
            
            image = ((image / 255.) - mean) / std
            image = image.transpose((2, 0, 1))
            gt = np.expand_dims(gt / 255., axis=0)
            
            images.append(image)
            gts.append(gt)
        '''
        return torch.tensor(np.array(images)).float().cuda(), torch.tensor(np.array(gts)).float().cuda()


    def __len__(self):
        return self.size

class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.images, self.gts = get_image_list(name, config, 'test')
        self.size = len(self.images)
        self.dataset_name = name

    def load_data(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if not self.config['orig_size']:
            image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = np.array(Image.open(self.gts[index]).convert('L'))
        name = self.images[index].split('/')[-1].split('.')[0]
        
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = torch.tensor(np.expand_dims(image, 0)).float()
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        
        return image, gt, name

def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()