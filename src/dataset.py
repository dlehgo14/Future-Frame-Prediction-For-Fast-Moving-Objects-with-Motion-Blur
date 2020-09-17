import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
import torch
import global_options

test_set_ratio = 0.1
n_obj = global_options.n_obj

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(self.dataroot)
    self.img = [os.path.join(self.dataroot, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img, img_name

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    # A
    images_A = sorted(os.listdir(os.path.join(self.dataroot, opts.phase + 'A')))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
    # B
    images_B = sorted(os.listdir(os.path.join(self.dataroot, opts.phase + 'B')))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.resize_x = opts.resize_size_x
    self.resize_y = opts.resize_size_y

    if opts.phase == 'train':
      transforms = [RandomCrop(opts.crop_size)]
    else:
      transforms = [CenterCrop(opts.crop_size)]
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      temp_b_index = random.randint(0, self.B_size - 1)
      data_B = self.load_img(self.B[temp_b_index], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):

    img = Image.open(img_name).convert('RGB')
    (w,h) = img.size
    if w < h:
        resize_x = self.resize_x
        resize_y = round(self.resize_x * h / w)
    else:
        resize_y = self.resize_y
        resize_x = round(self.resize_y * w / h)
    resize_img = Compose([Resize((resize_y, resize_x), Image.BICUBIC)])
    img = resize_img(img)
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

class dataset_pair(data.Dataset):
  def __init__(self, opts, blur_root, sharp_root, sharp_start_root, gt_root, gt_pos_root, test_mode=False):
    blur_datasets = np.load(blur_root)
    sharp_datasets = np.load(sharp_root)
    sharp_start_datasets = np.load(sharp_start_root)
    gt_datasets = np.load(gt_root)
    gt_pos_datasets = np.load(gt_pos_root)
    self.test_mode = test_mode

    category = blur_datasets.files[0]
    self.blur = blur_datasets[category]
    self.sharp = sharp_datasets[category]
    self.sharp_start = sharp_start_datasets[category]
    self.gt = gt_datasets[category]
    self.gt_pos = gt_pos_datasets[category]

    self.batch_size = self.blur.shape[0]
    self.train_set_num = int( (1-test_set_ratio) * self.batch_size )
    if not test_mode:
        self.blur = self.blur[:self.train_set_num]
        self.sharp = self.sharp[:self.train_set_num]
        self.sharp_start = self.sharp_start[:self.train_set_num]
        self.gt = self.gt[:self.train_set_num]
        self.gt_pos = self.gt_pos[:self.train_set_num]
    else:
        self.blur = self.blur[self.train_set_num:]
        self.sharp = self.sharp[self.train_set_num:]
        self.sharp_start = self.sharp_start[self.train_set_num:]
        self.gt = self.gt[self.train_set_num:]
        self.gt_pos = self.gt_pos[self.train_set_num:]
    # flatten
    self.blur = np.concatenate(self.blur,0)
    self.sharp = np.concatenate(self.sharp,0)
    self.sharp_start = np.concatenate(self.sharp_start,0)
    #self.gt = self.gt[:,:,3:,:]
    #self.gt = self.gt[:,:,::4,:] # [data_num, frame_num, ratio(16 -> 4), 2]
    # gt 평균 내서, 하나의 값으로 만들기
    #self.gt = self.gt.mean(2, keepdims=True)
    print(self.gt.shape)

    self.gt = np.concatenate(self.gt,0)
    self.gt = np.reshape(self.gt, [self.gt.shape[0],-1])
    self.gt = self.gt.astype(np.float32)

    #self.gt_pos = self.gt_pos[:,:,15:,:]
    self.gt_pos = np.concatenate(self.gt_pos,0)
    self.gt_pos = np.reshape(self.gt_pos, [self.gt_pos.shape[0],-1])
    self.gt_pos = self.gt_pos.astype(np.float32)

    self.dataset_size = len(self.blur)

    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.resize_x = opts.resize_size_x
    self.resize_y = opts.resize_size_y

    if opts.phase == 'train':
      transforms = [RandomCrop(opts.crop_size)]
    else:
      transforms = [CenterCrop(opts.crop_size)]
    #if not opts.no_flip:
    #  transforms.append(RandomHorizontalFlip())

    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('train A, B: %d images'%(self.dataset_size))
    return

  def __getitem__(self, index):
      if(index == (self.dataset_size-1)):
        index_2 = 0
        return self.__getitem__(index_2)
      else:
        index_2 = index+1

      if(index_2 == (self.dataset_size-1)):
        index_3 = 0
        return self.__getitem__(index_3)
      else:
        index_3 = index_2+1

      sharp_prev = self.load_img(self.sharp_start[index], self.input_dim_A)
      data_prev = self.load_img(self.blur[index], self.input_dim_B)
      data_post = self.load_img(self.blur[index_3], self.input_dim_B)
      data_A = self.load_img(self.sharp_start[index_2], self.input_dim_A)
      data_B = self.load_img(self.blur[index_2], self.input_dim_B)
      data_gt = self.gt[index_2]
      data_gt_pos = self.gt_pos[index_2]
      data_random = self.load_img(self.sharp_start[random.randint(0, self.dataset_size - 1)], self.input_dim_A)
      data_A_end = self.load_img(self.sharp[index_2], self.input_dim_A)

      if self.test_mode:
        return sharp_prev, data_prev, data_post, data_A, data_B, data_gt, data_gt_pos, data_A_end, data_random
      else:
        return sharp_prev, data_prev, data_post, data_A, data_B, data_gt, data_gt_pos, data_A_end

  def load_img(self, img, input_dim):
    (w,h,c)= img.shape
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    if w < h:
        resize_x = self.resize_x
        resize_y = round(self.resize_x * h / w)
    else:
        resize_y = self.resize_y
        resize_x = round(self.resize_y * w / h)
    resize_img = Compose([Resize((resize_y, resize_x), Image.BICUBIC)])
    img = resize_img(img)
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

class dataset_pair_group(data.Dataset):
  def __init__(self, opts, blur_root, sharp_root, sharp_start_root, gt_root, gt_pos_root, shutter_speed_root, test_split=False, origin_root = None):
    blur_datasets = np.load(blur_root)
    sharp_datsets = np.load(sharp_root)
    sharp_start_datasets = np.load(sharp_start_root)
    gt_datasets = []
    gt_pos_datasets = []
    for i in range(n_obj):
        if i == 0:
            gt_datasets.append(np.load(gt_root))
            gt_pos_datasets.append(np.load(gt_pos_root))
        else:
            gt_datasets.append(np.load(gt_root[:31] + str(i+1) + gt_root[31:]))
            gt_pos_datasets.append(np.load(gt_pos_root[:31]+str(i+1)+gt_pos_root[31:]))
    shutter_speed_datasets = np.load(shutter_speed_root)

    origin_version = False
    if( origin_root != None):
        origin_datasets = np.load(origin_root)
        origin_version = True
    self.origin_version = origin_version

    category = blur_datasets.files[0]
    self.blur = blur_datasets[category]
    self.sharp = sharp_datsets[category]
    self.sharp_start = sharp_start_datasets[category]
    self.gt = []
    self.gt_pos = []
    for i in range(n_obj):
        self.gt.append(gt_datasets[i][category])
        self.gt_pos.append(gt_pos_datasets[i][category])
    self.shutter_speed = shutter_speed_datasets[category]

    self.batch_size = self.blur.shape[0]
    self.train_set_num = int( (1-test_set_ratio) * self.batch_size )

    if not test_split:
        pass
        '''
        self.blur = self.blur[:self.train_set_num] # [data_num, sequence_num, h, w, 3]
        self.sharp = self.sharp[:self.train_set_num]
        self.sharp_start = self.sharp_start[:self.train_set_num]
        for i in range(n_obj):
            self.gt[i] = self.gt[i][:self.train_set_num]
            self.gt_pos[i] = self.gt_pos[i][:self.train_set_num]
        self.shutter_speed = self.shutter_speed[:self.train_set_num]
        '''
    else:
        #self.train_set_num = 64
        self.blur = self.blur[self.train_set_num:] # [data_num, sequence_num, h, w, 3]
        self.sharp = self.sharp[self.train_set_num:]
        self.sharp_start = self.sharp_start[self.train_set_num:]
        for i in range(n_obj):
            self.gt[i] = self.gt[i][self.train_set_num:]
            self.gt_pos[i] = self.gt_pos[i][self.train_set_num:]
        self.shutter_speed = self.shutter_speed[self.train_set_num:]

    if(origin_version):
        self.origin = origin_datasets[category]
        if test_split:
            self.origin = self.origin[self.train_set_num:] # only used in test mode
    # gt 갯수 조절
    #self.gt = self.gt[:,:,3:,:]
    #self.gt = self.gt[:,:,::4,:] # [data_num, frame_num, ratio(16 -> 4), 2]
    # gt 평균 내서, 하나의 값으로 만들기
    #self.gt = self.gt.mean(2, keepdims=True)
    print(self.gt[0].shape)

    for i in range(n_obj):
        self.gt[i] = np.reshape(self.gt[i], [self.gt[i].shape[0],self.gt[i].shape[1],-1])
        self.gt[i] = self.gt[i].astype(np.float32)

    #self.gt_pos = self.gt_pos[:,:,15:,:]
    for i in range(n_obj):
        self.gt_pos[i] = np.reshape(self.gt_pos[i], [self.gt_pos[i].shape[0],self.gt_pos[i].shape[1],-1])
        self.gt_pos[i] = self.gt_pos[i].astype(np.float32)

    self.shutter_speed = self.shutter_speed.astype(np.float32)

    self.dataset_size = len(self.blur)

    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.resize_x = opts.resize_size_x
    self.resize_y = opts.resize_size_y

    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('train A, B: %d images'%(self.dataset_size))
    return

  def __getitem__(self, index):
        data_A = self.load_img(self.sharp[index], self.input_dim_A)
        data_B = self.load_img(self.blur[index], self.input_dim_B)
        data_A_start = self.load_img(self.sharp_start[index], self.input_dim_B)
        data_gt = []
        data_gt_pos = []
        for i in range(n_obj):
            data_gt.append(self.gt[i][index])# * 10
            data_gt_pos.append(self.gt_pos[i][index])
        data_shutter_speed = self.shutter_speed[index]
        if(self.origin_version):
            data_origin = self.load_img(self.origin[index], 3)
            return data_A, data_B, data_gt, data_gt_pos, data_A_start, data_shutter_speed, data_origin
        return data_A, data_B, data_gt, data_gt_pos, data_A_start, data_shutter_speed

  def load_img(self, img_sequence, input_dim):
    (s,h,w,c)= img_sequence.shape
    ret = torch.zeros([s,c,self.resize_y,self.resize_x], dtype=torch.uint8).type('torch.FloatTensor')
    for n in range(s):
        img = Image.fromarray(img_sequence[n].astype('uint8'), 'RGB')
        resize_x = self.resize_x
        resize_y = self.resize_y
        resize_img = Compose([Resize((resize_y, resize_x), Image.BICUBIC)])
        img = resize_img(img)
        img = self.transforms(img)

        if input_dim == 1:
          img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
          img = img.unsqueeze(0)
        ret[n] = img


    return ret

  def __len__(self):
    return self.dataset_size

class dataset_pair_group_simple(data.Dataset):
  def __init__(self, opts, blur_root, sharp_root, sharp_start_root, gt_root, gt_pos_root, shutter_speed_root, test_mode=False, origin_root = None, sharp_whole_root = None):
    blur_datasets = np.load(blur_root)
    gt_datasets = []
    gt_pos_datasets = []
    for i in range(n_obj):
        if i == 0:
            gt_datasets.append(np.load(gt_root))
            gt_pos_datasets.append(np.load(gt_pos_root))
        else:
            gt_datasets.append(np.load(gt_root[:31] + str(i+1) + gt_root[31:]))
            gt_pos_datasets.append(np.load(gt_pos_root[:31]+str(i+1)+gt_pos_root[31:]))
    shutter_speed_datasets = np.load(shutter_speed_root)

    origin_version = False
    if( origin_root != None):
        origin_datasets = np.load(origin_root)
        origin_version = True
    sharp_whole_version = False
    if( sharp_whole_root != None):
        sharp_whole_datasets = np.load(sharp_whole_root)
        sharp_whole_version = True
    else:
        sharp_datasets = np.load(sharp_root)

    self.origin_version = origin_version
    self.sharp_whole_version = sharp_whole_version

    category = blur_datasets.files[0]
    self.blur = blur_datasets[category]
    self.gt = []
    self.gt_pos = []
    for i in range(n_obj):
        self.gt.append(gt_datasets[i][category])
        self.gt_pos.append(gt_pos_datasets[i][category])
    self.shutter_speed = shutter_speed_datasets[category]
    if(sharp_whole_version):
        self.sharp_whole = sharp_whole_datasets[category]
    else:
        self.sharp = sharp_datasets[category]

    self.batch_size = self.blur.shape[0]
    self.train_set_num = int( (1-test_set_ratio) * self.batch_size )

    if not test_mode:
        #self.train_set_num = 64
        self.blur = self.blur[:self.train_set_num] # [data_num, sequence_num, h, w, 3]
        for i in range(n_obj):
            self.gt[i] = self.gt[i][:self.train_set_num]
            self.gt_pos[i] = self.gt_pos[i][:self.train_set_num]
        self.shutter_speed = self.shutter_speed[:self.train_set_num]
        if(sharp_whole_version):
            self.sharp_whole = self.sharp_whole[:self.train_set_num]
        else:
            self.sharp = self.sharp[:self.train_set_num]
    else:
        #self.train_set_num = 64
        self.blur = self.blur[self.train_set_num:] # [data_num, sequence_num, h, w, 3]
        for i in range(n_obj):
            self.gt[i] = self.gt[i][self.train_set_num:]
            self.gt_pos[i] = self.gt_pos[i][self.train_set_num:]
        self.shutter_speed = self.shutter_speed[self.train_set_num:]
        if(sharp_whole_version):
            self.sharp_whole = self.sharp_whole[self.train_set_num:]
        else:
            self.sharp = self.sharp[self.train_set_num:]

    if(origin_version):
        self.origin = origin_datasets[category]
        self.origin = self.origin[self.train_set_num:] # only used in test mode
    # gt 갯수 조절
    #self.gt = self.gt[:,:,3:,:]
    #self.gt = self.gt[:,:,::4,:] # [data_num, frame_num, ratio(16 -> 4), 2]
    # gt 평균 내서, 하나의 값으로 만들기
    #self.gt = self.gt.mean(2, keepdims=True)
    print(self.gt[0].shape)

    for i in range(n_obj):
        self.gt[i] = np.reshape(self.gt[i], [self.gt[i].shape[0],self.gt[i].shape[1],-1])
        self.gt[i] = self.gt[i].astype(np.float32)

    #self.gt_pos = self.gt_pos[:,:,15:,:]
    for i in range(n_obj):
        self.gt_pos[i] = np.reshape(self.gt_pos[i], [self.gt_pos[i].shape[0],self.gt_pos[i].shape[1],-1])
        self.gt_pos[i] = self.gt_pos[i].astype(np.float32)

    self.shutter_speed = self.shutter_speed.astype(np.float32)

    self.dataset_size = len(self.blur)

    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.resize_x = opts.resize_size_x
    self.resize_y = opts.resize_size_y

    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('train A, B: %d images'%(self.dataset_size))
    return

  def __getitem__(self, index):
        data_B = self.load_img(self.blur[index], self.input_dim_B)
        data_gt = []
        data_gt_pos = []
        for i in range(n_obj):
            data_gt.append(self.gt[i][index])# * 10
            data_gt_pos.append(self.gt_pos[i][index])
        data_shutter_speed = self.shutter_speed[index]
        if(self.sharp_whole_version):
            data_S_whole = self.load_img(self.sharp_whole[index], self.input_dim_B)
            return data_B, data_gt, data_gt_pos, data_shutter_speed, data_S_whole, torch.tensor(0)
        data_A = self.load_img(self.sharp[index], self.input_dim_A)

        return data_B, data_gt, data_gt_pos, data_shutter_speed, torch.tensor(0), data_A

  def load_img(self, img_sequence, input_dim):
    (s,h,w,c)= img_sequence.shape
    ret = torch.zeros([s,c,self.resize_y,self.resize_x], dtype=torch.uint8).type('torch.FloatTensor')
    for n in range(s):
        img = Image.fromarray(img_sequence[n].astype('uint8'), 'RGB')
        resize_x = self.resize_x
        resize_y = self.resize_y
        resize_img = Compose([Resize((resize_y, resize_x), Image.BICUBIC)])
        img = resize_img(img)
        img = self.transforms(img)

        if input_dim == 1:
          img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
          img = img.unsqueeze(0)
        ret[n] = img


    return ret

  def __len__(self):
    return self.dataset_size
