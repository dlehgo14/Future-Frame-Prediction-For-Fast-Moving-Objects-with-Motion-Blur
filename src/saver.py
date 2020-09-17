import os
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, Compose
import torch
import global_options

one_by_one_save_mode = global_options.one_by_one_save_mode
# tensor to PIL Image
def tensor2img(img):
  img = img[0].cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (np.transpose(img, (1, 2, 0))*0.5+0.5) * 255.0
  return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path, yuv=False):
    if not os.path.exists(path):
        os.mkdir(path)
    img = tensor2img(imgs)
    img = Image.fromarray(img)
    img.save(os.path.join(path, names))

class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.visualize_dir = os.path.join(opts.visualize_root, 'prediction')
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq
    self.opts = opts
    # make directory
    if not os.path.exists(self.display_dir):
      os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
        assembled_images = model.assemble_outputs()
        img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
        torchvision.utils.save_image(assembled_images*0.5+0.5, img_filename, nrow=1)
    elif ep == -1:
        assembled_images = model.assemble_outputs()
        img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
        torchvision.utils.save_image(assembled_images*0.5+0.5, img_filename, nrow=1)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
      model.save('%s/last.pth' % self.model_dir, ep, total_it)

  def write_pred_img(self, ep, n, input_n, ret, ret_gt, ret_origin = None, origin_version = False):
    # make dir
    try:
        if not(os.path.isdir(self.visualize_dir)):
            os.makedirs(os.path.join(self.visualize_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    # interval
    interval = torch.ones(self.opts.resize_size_y*10*3).reshape([1,3,self.opts.resize_size_y,10]).detach().cuda(self.opts.gpu)
    interval_h = torch.ones(3*10*(10+self.opts.resize_size_x)*ret.size()[1]) \
    .reshape([1,3,10,(10+self.opts.resize_size_x)*ret.size()[1]]).detach().cuda(self.opts.gpu)
    # main
    ret = ret.detach().cuda(self.opts.gpu)
    ret_gt = ret_gt.detach().cuda(self.opts.gpu)
    if (origin_version):
        ret_origin = ret_origin.detach().cuda(self.opts.gpu)

    images = torch.tensor([]).cuda(self.opts.gpu)
    for i in range(ret.size()[0]):
        imgs = ret[i]
        row = torch.tensor([]).cuda(self.opts.gpu)
        for k in range(imgs.size()[0]):
            row = torch.cat( (row, interval, imgs[k:k+1,::]), 3)
            if one_by_one_save_mode:
                # make dir
                folder_dir = '%s/gen_%05d_%02d_%03d_%01d' % (self.visualize_dir, ep, input_n,n,i)
                try:
                    if not(os.path.isdir(folder_dir)):
                        os.makedirs(os.path.join(folder_dir))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        print("Failed to create directory!!!!!")
                        raise
                img_filename = folder_dir+ "/%02d.jpg" % ( k)
                torchvision.utils.save_image(imgs[k:k+1,::]*0.5+0.5, img_filename, nrow=1)
        images = torch.cat( (images, interval_h, row), 2)

        imgs = ret_gt[i]
        row = torch.tensor([]).cuda(self.opts.gpu)
        for k in range(imgs.size()[0]):
            row = torch.cat( (row, interval, imgs[k:k+1,::]), 3)
            if one_by_one_save_mode:
                # make dir
                folder_dir = '%s/gen_%05d_%02d_%03d_%01d_gt' % (self.visualize_dir, ep, input_n,n,i)
                try:
                    if not(os.path.isdir(folder_dir)):
                        os.makedirs(os.path.join(folder_dir))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        print("Failed to create directory!!!!!")
                        raise
                img_filename = folder_dir+ "/%02d.jpg" % ( k)
                torchvision.utils.save_image(imgs[k:k+1,::]*0.5+0.5, img_filename, nrow=1)
        images = torch.cat( (images, interval_h, row), 2)

        if (origin_version):
            imgs = ret_origin[i]
            row = torch.tensor([]).cuda(self.opts.gpu)
            for k in range(imgs.size()[0]):
                row = torch.cat( (row, interval, imgs[k:k+1,::]), 3)
            images = torch.cat( (images, interval_h, row), 2)

    img_filename = '%s/gen_%05d_%02d_%03d.jpg' % (self.visualize_dir, ep, input_n, n)
    torchvision.utils.save_image(images*0.5+0.5, img_filename, nrow=1)

    # gif
    img = ret[0].cpu().float().numpy()
    img = (np.transpose(img, (0, 2, 3, 1))*0.5+0.5) * 255.0
    img = img.astype(np.uint8)
    print(img.shape)
    gif = []
    for i in img:
        gif.append(Image.fromarray(i))
    gif[0].save('%s/gif_%05d_%02d_%03d.gif' % (self.visualize_dir, ep, input_n, n), format='GIF', append_images=gif[1:], save_all=True, duration=100, loop=0)
