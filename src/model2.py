import torch
import torch.nn as nn
import time
from torchvision import transforms
import global_options
from pytorch_ssim import pytorch_ssim
import pytorch_msssim

network2_mode = global_options.network2_mode
n_obj = global_options.n_obj
pred_relation_mode = global_options.pred_relation_mode

if network2_mode:
    print("network2 mode ON")
    import networks_179 as networks
else:
    import networks


pred_mode =global_options.pred_mode
gen_mode = global_options.gen_mode
vel_secret_mode = global_options.vel_secret_mode
encoder_tuning_mode = global_options.encoder_tuning_mode
sharp_whole_mode = global_options.sharp_whole_mode
gen_manual_mode = global_options.gen_manual_mode
z_delete_mode = global_options.z_delete_mode
sharp_loss_mode = global_options.sharp_loss_mode
sharp_gen_mode = global_options.sharp_gen_mode
last_vel_loss_mode = global_options.last_vel_loss_mode
dir_mode = global_options.dir_mode
throwing_mode = global_options.throwing_mode
# content 지운 버전

class UID(nn.Module):
  def __init__(self, opts):
    super(UID, self).__init__()
    self.pred_mode =pred_mode
    self.gen_mode = gen_mode
    # transforms for pic save
    #self.size_transform = transforms.Compose([transforms.Scale((128,128))])
    self.ssim_loss = pytorch_ssim.SSIM(window_size = 7)
    # parameters
    lr = opts.lr
    self.opts = opts
    self.nz = 32
    self.ratio = self.nz//2
    self.width = opts.resize_size_x
    self.height = opts.resize_size_y

    # interval for saving examples
    self.interval = torch.ones(opts.resize_size_y*10*3).reshape([1,3,opts.resize_size_y,10]).detach().cuda(self.opts.gpu)


    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b, norm_layer=networks.LayerNorm, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    self.enc_a = networks.E_attr_concat(opts.input_dim_b, self.nz,
          norm_layer=networks.LayerNorm, nl_layer=networks.get_non_linearity(layer_type='lrelu'))

    # prediction network
    self.enc_pred = networks.E_pred(opts.input_dim_a, self.nz, norm_layer=networks.LayerNorm, nl_layer=networks.get_non_linearity(layer_type='lrelu'), h = opts.resize_size_y, w = opts.resize_size_x, gpu=opts.gpu)
    self.enc_relation = networks.E_pred_relation(norm_layer=networks.LayerNorm, nl_layer=networks.get_non_linearity(layer_type='lrelu'), h = opts.resize_size_y, w = opts.resize_size_x, gpu=opts.gpu)

    # generator
    if gen_manual_mode:
        self.gen = networks.G_manual(nz=self.nz, gpu= opts.gpu, h =opts.resize_size_y, w = opts.resize_size_x)
    else:
        self.gen = networks.G_concat(opts.input_dim_a, nz=self.nz, h =opts.resize_size_y, w = opts.resize_size_x, gpu= opts.gpu)
    if sharp_gen_mode:
        self.gen_sharp = networks.G_sharp(opts.input_dim_a, nz=self.nz, h =opts.resize_size_y, w = opts.resize_size_x)

    # optimizers
    if vel_secret_mode:
        self.enc_pred_opt = torch.optim.Adam(self.enc_pred.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        if pred_relation_mode:
            self.enc_relation_opt = torch.optim.Adam(self.enc_relation.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        if not gen_manual_mode:
            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0, betas=(0.5, 0.999), weight_decay=0.0001)
        if sharp_gen_mode:
            self.gen_sharp_opt = torch.optim.Adam(self.gen_sharp.parameters(), lr=0, betas=(0.5, 0.999), weight_decay=0.0001)
    else:
        self.enc_pred_opt = torch.optim.Adam(self.enc_pred.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        if pred_relation_mode:
            self.enc_relation_opt = torch.optim.Adam(self.enc_relation.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        if not gen_manual_mode:
            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        if sharp_gen_mode:
            self.gen_sharp_opt = torch.optim.Adam(self.gen_sharp.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # freeze some networks
    if vel_secret_mode:
        for name, p in self.enc_pred.named_parameters():
            if "fc." in name:
                p.requires_grad = False
                print(name + " in enc_pred trainable False")
            elif "conv." in name:
                p.requires_grad = False
                print(name + " in enc_pred trainable False")
    if encoder_tuning_mode:
        print("##### encoder fine tuning mode: fix the convs of encoders #####")
        for param in self.enc_c.conv_B.parameters():
            param.requires_grad = False
        for param in self.enc_a.conv_dir.parameters():
            param.requires_grad = False
        for param in self.enc_a.conv_B.parameters():
            param.requires_grad = False

  def initialize(self):

    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)
    self.enc_pred.apply(networks.gaussian_weights_init)
    if pred_relation_mode:
        self.enc_relation.apply(networks.gaussian_weights_init)
    if not gen_manual_mode:
        self.gen.apply(networks.gaussian_weights_init)
    if sharp_gen_mode:
        self.gen_sharp.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):

    self.enc_pred_sch = networks.get_scheduler(self.enc_pred_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    if pred_relation_mode:
        self.enc_relation_sch = networks.get_scheduler(self.enc_relation_opt, opts, last_ep)
    if not gen_manual_mode:
        self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)
    if sharp_gen_mode:
        self.gen_sharp_sch = networks.get_scheduler(self.gen_sharp_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu

    # multi-gpu
    if (self.opts.multi_gpu) and (torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.enc_pred = nn.DataParallel(self.enc_pred)
        if pred_relation_mode:
            self.enc_relation = nn.DataParallel(self.enc_relation)
        self.enc_c = nn.DataParallel(self.enc_c)
        self.enc_a = nn.DataParallel(self.enc_a)
        if not gen_manual_mode:
            self.gen = nn.DataParallel(self.gen)
        if sharp_gen_mode:
            self.gen_sharp = nn.DataParallel(self.gen_sharp)

    self.enc_pred.cuda(self.gpu)
    if pred_relation_mode:
        self.enc_relation.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    if not gen_manual_mode:
        self.gen.cuda(self.gpu)
    if sharp_gen_mode:
        self.gen_sharp.cuda(self.gpu)


  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  # transformation functions
  def translate(self, img, translate_val):
    ret = img
    return ret
  def rotate(self, img, rotate_val):
    ret = img
    return ret
  def scale(self, img, scale_val):
    ret = img
    return ret
  def rotate_vector(self, vel, rotate_val):
    ret = vel
    return ret

  def forward(self, train=True):
    # for transform

    self.batch_size = self.input_B.size(0)
    self.zero_vel = torch.zeros([self.batch_size,self.nz]).cuda(self.gpu)
    self.full_shutter_speed = torch.ones([self.batch_size,1]).cuda(self.gpu)
    '''
    self.translate_val = self.get_z_random(self.batch_size, 2)
    self.scale_val = self.get_z_random(self.batch_size, 2)
    self.rotate_val = self.get_z_random(self.batch_size, 2)
    self.traslate_B = self.translate(self.target_B, self.translate_val)
    self.scale_B = self.scale(self.target_B, self.scale_val)
    self.rotate_B = self.rotate(self.target_B, self.rotate_val)
    '''
    # gt inverse
    self.gt_inverse = []
    input_B = self.input_B.clone()
    self.mag = []
    self.dir = []
    self.vel_pred = []
    self.vel_pred_inverse = []
    self.pos_pred = []
    self.fake_B_encoded = []
    self.fake_B_encoded_with_gt = []
    # ground truth
    self.gt_vel = []
    self.gt_pos = []
    # pred mode
    self.next_vel_pred = []
    self.next_pos_pred = []
    self.next_gt_vel_real = []
    self.next_gt_pos_real = []
    input_gt_vel = [] # [o, n, 32]
    input_gt_pos = []
    self.post_B_pred = []
    if vel_secret_mode or encoder_tuning_mode:
        self.next_pos_encoded = []
        self.next_vel_encoded = []
    for o in range(n_obj):
        input_gt_vel.append(self.input_gt_vel[o].clone())
        input_gt_pos.append(self.input_gt_pos[o].clone())

    # encoder
    for o in range(n_obj):
        # pick min_o whose position is higher (if the height is same, pick left side one)
        gt_vel = input_gt_vel[0].clone() # [n, 32]
        gt_pos = input_gt_pos[0].clone() # [n, 2]
        next_gt_vel = self.next_gt_vel[0].clone()
        next_gt_pos = self.next_gt_pos[0].clone()
        n = 0 # batch num
        for batch in gt_vel:
            min_o = 0
            for oo in range(1, n_obj):
                if gt_pos[n][1] < input_gt_pos[oo][n][1]: # pick higher one
                    gt_vel[n] = input_gt_vel[oo][n].clone()
                    gt_pos[n] = input_gt_pos[oo][n].clone()
                    if pred_mode:
                        next_gt_vel[n] = self.next_gt_vel[oo][n].clone()
                        next_gt_pos[n] = self.next_gt_pos[oo][n].clone()
                    min_o = oo
                elif gt_pos[n][1] == input_gt_pos[oo][n][1]:
                    if gt_pos[n][0] > input_gt_pos[oo][n][0]: # pick left side one
                        gt_vel[n] = input_gt_vel[oo][n].clone()
                        gt_pos[n] = input_gt_pos[oo][n].clone()
                        if pred_mode:
                            next_gt_vel[n] = self.next_gt_vel[oo][n].clone()
                            next_gt_pos[n] = self.next_gt_pos[oo][n].clone()
                        min_o = oo
            input_gt_pos[min_o][n] = -100
            n+=1
        self.gt_vel.append(gt_vel)
        self.gt_pos.append(gt_pos)
        self.next_gt_vel_real.append(next_gt_vel)
        self.next_gt_pos_real.append(next_gt_pos)

        self.gt_inverse.append(-gt_vel.view([-1,self.nz//2,2]).flip(1).view([-1,self.nz]))

        # position encode
        if (type(self.given_vel) == type(None)):
            pos_pred = self.enc_c.forward(self.prev_B, input_B, forward_b = True)
        else:
            pos_pred = self.given_pos[o]
        self.pos_pred.append(pos_pred)

        # get vel
        if dir_mode:
            mag, dir = self.enc_a.forward(self.prev_B, input_B)
            self.mag.append(mag)
            self.dir.append(dir)
            if (type(self.given_vel) == type(None)):
                vel_pred = mag.mul(dir.mul(2).sub(1))
            else:
                vel_pred = self.given_vel[o]
            self.vel_pred.append(vel_pred)
            self.vel_pred_inverse.append(-vel_pred.view([-1,self.nz//2,2]).flip(1).view([-1,self.nz]))

            # get inverse vel
            #self.mag_inverse, self.dir_inverse = self.enc_a.forward(self.post_B, self.input_B)
            #self.vel_inverse = self.mag_inverse.mul(self.dir_inverse.mul(2).sub(1))
        else:
            self.vel_pred = self.enc_a.forward(self.prev_B, input_B)
            self.vel_pred_inverse = -self.vel_pred.view([-1,self.nz//2,2]).flip(1).view([-1,self.nz])
            self.vel_inverse = self.enc_a.forward(self.post_B, input_B)

        if(gen_mode):
            #self.fake_B_encoded = self.gen.forward(self.input_gt_pos, self.vel_pred, self.shutter_speed)
            self.fake_B_encoded_with_gt.append(self.gen.forward(gt_pos, gt_vel, self.shutter_speed))
            self.fake_B_encoded.append(self.gen.forward(self.pos_pred[o], self.vel_pred[o], self.shutter_speed))
            input_B = input_B - self.fake_B_encoded_with_gt[o].clone() - 1 #[-1, 1]
            # get reconstructed vel
            '''
            if dir_mode:
                self.mag_recons, self.dir_recons = self.enc_a.forward(self.prev_B, self.fake_B_encoded_with_gt)
                self.vel_recons = self.mag_recons.mul(self.dir_recons.mul(2).sub(1))
            '''
        if vel_secret_mode or encoder_tuning_mode:
            self.next_pos_encoded.append(self.enc_c.forward(self.input_B, self.post_B, forward_b = True))
            mag, dir = self.enc_a.forward(self.input_B, self.post_B)
            self.next_vel_encoded.append(mag.mul(dir.mul(2).sub(1)))

    # pred
    if pred_mode:
        for o in range(n_obj):
            if train:
                if not vel_secret_mode:
                    _, next_vel_pred, _, self.z_next = self.enc_pred.forward(None, self.gt_pos[o], self.gt_vel[o], None)
                else:
                    _, next_vel_pred, _, self.z_next = self.enc_pred.forward(None, self.pos_pred[o], self.vel_pred[o], None)

                # relation mode
                if pred_relation_mode:
                    for oo in range(n_obj):
                        if o == oo:
                            continue
                        next_vel_pred = self.enc_relation.forward(next_vel_pred, self.gt_vel[oo], self.gt_pos[o], self.gt_pos[oo])
            else:
                if (type(self.given_vel) == type(None)):
                    next_vel_pred, _, self.z_next = self.enc_pred.forward(pos = self.pos_pred[o], vel = self.vel_pred[o], z = None, forward_future = True)
                    # relation mode
                    if pred_relation_mode:
                        for oo in range(n_obj):
                            if o == oo:
                                continue
                            next_vel_pred = self.enc_relation.forward(next_vel_pred, self.vel_pred[oo], self.pos_pred[o], self.pos_pred[oo])

                else:
                    next_vel_pred, _, self.z_next = self.enc_pred.forward(pos = self.given_pos[o], vel = self.given_vel[o], z = self.z, forward_future = True)
                    # relation mode
                    if pred_relation_mode:
                        for oo in range(n_obj):
                            if o == oo:
                                continue
                            next_vel_pred = self.enc_relation.forward(next_vel_pred, self.given_vel[oo], self.given_pos[o], self.given_pos[oo])

            self.next_vel_pred.append(next_vel_pred)

            next_pos_preds = torch.zeros([self.batch_size, self.ratio, 2]).cuda(self.gpu)
            if (type(self.given_vel) == type(None)):
                cur_pos = self.pos_pred[o].clone()
            else:
                cur_pos = self.given_pos[o].clone()
            vel = self.next_vel_pred[o].view(-1,16,2)
            for i in range(self.ratio):
                cur_pos = cur_pos + vel[:,i]
                next_pos_preds [:,i] = cur_pos.clone()
            next_pos_preds = next_pos_preds.view(-1,self.nz)
            next_pos_pred = self.pos_pred[o] + torch.sum(self.next_vel_pred[o].view(-1,16,2),1)*0.01
            self.next_pos_pred.append(next_pos_pred)




    # gen and pred
    # next frame prediction
    if(pred_mode and gen_mode and (not train)) or vel_secret_mode:
        for o in range(n_obj):
            post_B_pred = self.gen.forward(self.next_pos_pred[o], self.next_vel_pred[o], self.shutter_speed)
            self.post_B_pred.append(post_B_pred)

        if(sharp_whole_mode and train):
            self.pos_pred_whole = torch.zeros([self.batch_size, self.ratio, 2]).cuda(self.gpu)
            # this is current pos version
            cur_pos = self.pos_pred
            '''
            vel = self.vel_pred.view(-1,16,2)
            self.pos_pred_whole[:,self.ratio - 1] = cur_pos
            for i in range(1,self.ratio):
                cur_pos = cur_pos - vel[:,self.ratio -i]
                self.pos_pred_whole [:,self.ratio - 1 - i] = cur_pos
            '''
            # this is next pos version
            vel = self.next_vel_pred.view(-1,16,2)
            for i in range(0,self.ratio):
                cur_pos = cur_pos + vel[:,i] * 0.01
                self.pos_pred_whole [:,i] = cur_pos
            #self.pos_pred_whole = self.pos_pred_whole.view(-1,self.nz)
            self.sharp_pred_whole = torch.zeros([self.batch_size, self.ratio, 3, self.height, self.width]).cuda(self.gpu)
            if (sharp_gen_mode):
                for i in range(self.ratio):
                    self.sharp_pred_whole[:,i] = self.gen_sharp.forward(self.pos_pred_whole[:,i])
            else:
                for i in range(self.ratio):
                    self.sharp_pred_whole[:,i] = self.gen.forward(self.pos_pred_whole[:,i], self.zero_vel, self.full_shutter_speed)

        if sharp_gen_mode:
            self.sample_sharp_gen = self.gen_sharp.forward(self.pos_pred[0])
        elif (not train) or (not sharp_whole_mode):
            self.sample_sharp_gen = self.gen.forward(self.pos_pred[0], self.zero_vel, self.full_shutter_speed)

    if gen_mode:
        self.recons_B = self.fake_B_encoded[0].clone()
        self.recons_B_with_gt = self.fake_B_encoded_with_gt[0].clone()
        for oo in range(1, n_obj):
            self.recons_B += (self.fake_B_encoded[oo] + 1)
            self.recons_B_with_gt += (self.fake_B_encoded_with_gt[oo] + 1)
        self.recons_B = torch.clamp(self.recons_B, min=-1.0, max=1.0)
        self.recons_B_with_gt = torch.clamp(self.recons_B_with_gt, min=-1.0, max=1.0)
        if (pred_mode and (not train)) or vel_secret_mode:
            self.post_B_pred_recons = self.post_B_pred[0].clone()
            for oo in range(1, n_obj):
                self.post_B_pred_recons += (self.post_B_pred[oo] + 1)
            self.post_B_pred_recons = torch.clamp(self.post_B_pred_recons, min=-1.0, max=1.0)



  '''
  def test(self, images_sharp_prev, image_prev, image_post, image_a, image_b, gt, gt_pos, images_a_end, data_random, next_gt = None, next_gt_pos = None, z = None, given_vel = None, given_pos = None, last_gt = None, left_steps=1, gt_pos_set = None, shutter_speed = 1):
    self.prev_S = images_sharp_prev
    self.prev_B = image_prev
    self.post_B = image_post
    self.input_S = image_a
    self.input_B = image_b
    self.input_gt_vel = gt
    self.input_gt_pos = gt_pos[:,30:]
    self.input_S_end = images_a_end
    self.data_random = data_random
    self.next_gt_vel = next_gt#[:,:2]
    self.next_gt_pos = next_gt_pos
    if type(next_gt_pos) != type(None):
        self.next_gt_pos = next_gt_pos[:,30:]
    #self.next_gt = gt[:,30:]
    self.z = z
    self.given_vel = given_vel # v2 -> v1 바로 대입
    self.given_pos = given_pos
    self.last_gt_vel = last_gt
    self.left_steps = left_steps
    if type(gt_pos_set) != type(None):
        self.gt_pos_set = gt_pos_set[:,:,30:]
    self.shutter_speed = shutter_speed
    self.forward(train=False)
    self.backward(train=False)
    self.backward_G_alone(train=False)

  def update(self, images_sharp_prev, image_prev, image_post, image_a, image_b, gt, gt_pos, images_a_end, next_gt = None, next_gt_pos = None, z = None, given_vel = None, given_pos = None, last_gt = None, left_steps=1, gt_pos_set = None, shutter_speed = 1):
    # inputs
    self.prev_S = images_sharp_prev
    self.prev_B = image_prev
    self.post_B = image_post
    self.input_S = image_a
    self.input_B = image_b
    self.input_gt_vel = gt
    self.input_gt_pos = gt_pos[:,30:]
    self.input_S_end = images_a_end

    self.next_gt_vel = next_gt#[:,:2]
    self.next_gt_pos = next_gt_pos
    if type(next_gt_pos) != type(None):
        self.next_gt_pos = next_gt_pos[:,30:]
    #self.next_gt = gt[:,30:]
    self.z = z
    self.given_vel = given_vel
    self.given_pos = given_pos
    self.last_gt_vel = last_gt
    self.left_steps = left_steps
    self.gt_pos_set = gt_pos_set[:,:,30:]
    self.shutter_speed = shutter_speed

    # forward
    self.forward()

    # update encoders and generator
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.enc_pred_opt.zero_grad()
    self.backward()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()
    self.enc_pred_opt.step()
    #self.theta_opt.step()

    # update content encoder and generator (gan loss)
    #self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone(train=False)
    #self.enc_c_opt.step()
    self.gen_opt.step()
  '''

  def test(self, image_prev, image_post, image_b, gt, gt_pos, next_gt = None, next_gt_pos = None, z = None, given_vel = None, given_pos = None, last_gt = None, left_steps=1, gt_pos_set = None, shutter_speed = 1):
    self.prev_B = image_prev
    self.post_B = image_post
    self.input_B = image_b
    self.input_gt_vel = []
    self.input_gt_pos = []
    # o, b, datasize
    for o in range(n_obj):
        self.input_gt_vel.append(gt[o])
        self.input_gt_pos.append(gt_pos[o][:,30:])

    if type(next_gt_pos) != type(None):
        self.next_gt_vel = []
        self.next_gt_pos = []
        self.whole_next_gt_pos = []
        for o in range(n_obj):
            self.next_gt_vel.append(next_gt[o])#[:,:2]
            self.next_gt_pos.append(next_gt_pos[o][:,30:])
            self.whole_next_gt_pos.append(next_gt_pos[o])
    else:
        self.next_gt_vel = next_gt #None
        self.next_gt_pos = next_gt_pos

    self.z = z
    self.given_vel = given_vel # v2 -> v1 바로 대입
    self.given_pos = given_pos
    self.last_gt_vel = last_gt
    self.left_steps = left_steps
    if type(gt_pos_set) != type(None):
        self.gt_pos_set = []
        for o in range(n_obj):
            self.gt_pos_set.append(gt_pos_set[o][:,:,30:])
    self.shutter_speed = shutter_speed
    self.forward(train=False)
    self.backward(train=False)
    self.backward_G_alone(train=False)

  def update(self, image_prev, image_post, image_b, gt, gt_pos, next_gt = None, next_gt_pos = None, z = None, given_vel = None, given_pos = None, last_gt = None, left_steps=1, gt_pos_set = None, shutter_speed = 1, images_s_whole = None, images_a = None):
    # inputs
    self.prev_B = image_prev
    self.post_B = image_post
    self.input_B = image_b
    self.input_gt_vel = []
    self.input_gt_pos = []

    for o in range(n_obj):
        self.input_gt_vel.append(gt[o])
        self.input_gt_pos.append(gt_pos[o][:,30:])

    if type(next_gt_pos) != type(None):
        self.next_gt_vel = []
        self.next_gt_pos = []
        self.whole_next_gt_pos = []
        for o in range(n_obj):
            self.next_gt_vel.append(next_gt[o])#[:,:2]
            self.next_gt_pos.append(next_gt_pos[o][:,30:])
            self.whole_next_gt_pos.append(next_gt_pos[o])
    else:
        self.next_gt_vel = next_gt #None
        self.next_gt_pos = next_gt_pos

    if(sharp_whole_mode):
        self.images_s_whole = images_s_whole
    else:
        self.images_a = images_a


    #self.next_gt = gt[:,30:]
    self.z = z
    self.given_vel = given_vel
    self.given_pos = given_pos
    self.last_gt_vel = last_gt
    self.left_steps = left_steps
    if type(gt_pos_set) != type(None):
        self.gt_pos_set = []
        for o in range(n_obj):
            self.gt_pos_set.append(gt_pos_set[o][:,:,30:])
    self.shutter_speed = shutter_speed

    # forward
    self.forward()

    # update encoders and generator
    # throwing mode면서 given_vel이 None이 아니면 enc, gen 고정
    if not throwing_mode or (type(self.given_vel) == type(None)):
        if (not vel_secret_mode):
            self.enc_c_opt.zero_grad()
            self.enc_a_opt.zero_grad()
        if (not vel_secret_mode) and (not gen_manual_mode) and (not encoder_tuning_mode):
            self.gen_opt.zero_grad()
            #pass
    if pred_mode and (not encoder_tuning_mode):
        self.enc_pred_opt.zero_grad()
        if pred_relation_mode:
            self.enc_relation_opt.zero_grad()
    if (sharp_gen_mode):
        self.gen_sharp_opt.zero_grad()
    self.backward()
    if not throwing_mode or (type(self.given_vel) == type(None)):
        if (not vel_secret_mode):
            self.enc_c_opt.step()
            self.enc_a_opt.step()
        if (not vel_secret_mode) and (not gen_manual_mode) and (not encoder_tuning_mode):
            self.gen_opt.step()
            #pass
    if pred_mode and (not encoder_tuning_mode):
        self.enc_pred_opt.step()
        if pred_relation_mode:
            self.enc_relation_opt.step()
    if (sharp_gen_mode):
        self.gen_sharp_opt.step()
    #self.theta_opt.step()

    # update content encoder and generator (gan loss)
    #self.enc_c_opt.zero_grad()
    #if not vel_secret_mode:
        #self.gen_opt.zero_grad()
    self.backward_G_alone(train=False)
    #self.enc_c_opt.step()
    #if not vel_secret_mode:
        #self.gen_opt.step()

  def backward(self, train = True):
    eps = 1e-4
    '''
    #================for vel_to_pos_loss===================
    v = self.vel_pred.view(-1,16,2)
    c = self.pos_pred
    vel_pred_to_pos = torch.zeros([self.vel_pred.size()[0], 16, 2]).cuda(self.gpu)
    for i in range(0,16):
        vel_pred_to_pos[:,15-i] = c
        c = c - v[:,15-i] * 0.01
    vel_pred_to_pos = vel_pred_to_pos.view(-1,32)

    v = self.input_gt_vel.view(-1,16,2)
    c = self.input_gt_pos
    vel_gt_to_pos = torch.zeros([self.vel_pred.size()[0], 16, 2]).cuda(self.gpu)
    for i in range(0,16):
        vel_gt_to_pos[:,15-i] = c
        c = c - v[:,15-i] * 0.01
    vel_gt_to_pos = vel_gt_to_pos.view(-1,32)

    vel_to_pos_loss = torch.mean( (vel_pred_to_pos - vel_gt_to_pos).pow(2) ) * 10
    #==========================================================
    '''
    # vel_secret_mode
    #loss_inverse_2 = torch.mean((self.vel_inverse - self.vel_pred_inverse).pow(2)) * 10
    if pred_mode:
        #loss_next_pos_2 = torch.min([torch.mean((self.next_pos_pred[o] - self.next_gt_pos_real[o]).pow(2)) for o in range(n_obj)]) * 100
        loss_next_pos_2 = -1
    else:
        loss_next_pos_2 = -1
    #loss_next_pos_2 = torch.mean((self.next_pos_preds - self.whole_next_gt_pos).pow(2)) * 100
    #loss_inverse_dir = -torch.mean(torch.log(self.dir_inverse+eps)*(1-self.dir) + torch.log(1-self.dir_inverse+eps)*(self.dir)) * 10


    # loss_vel: direction * magnitude

    # pick min_o that minimizes the velocity error
    '''
    self.gt_vel = self.input_gt_vel[0].clone()
    self.gt_pos = self.input_gt_pos[0].clone()
    loss_vel = torch.mean( (self.vel_pred - self.gt_vel).pow(2), dim=[-1])
    for o in range(1,n_obj):
        n = 0
        comp = torch.mean( (self.input_gt_vel[o] - self.gt_vel).pow(2), dim=[-1])
        for batch in self.gt_vel:
            if loss_vel[n] > comp[n]:
                loss_vel[n] = comp[n]
                self.gt_vel[n] = self.input_gt_vel[o][n]
                self.gt_pos[n] = self.input_gt_pos[o][n]
            n+=1
    loss_vel = torch.mean(loss_vel) * 10
    '''

    loss_vel_dir = []
    loss_dir_entropy = []
    loss_vel_speed = []
    loss_pos = []
    loss_vel = []
    loss_pred_vel = []
    if vel_secret_mode or encoder_tuning_mode:
        loss_next_pos_encoded = []
        loss_next_vel_encoded = []

    for oo in range(n_obj):

        loss_vel.append(torch.mean( (self.vel_pred[oo] - self.gt_vel[oo]).pow(2)) * 10)
        loss_pos.append(torch.mean((self.pos_pred[oo] - self.gt_pos[oo]).pow(2)) * 10)
        if vel_secret_mode or encoder_tuning_mode:
            loss_next_pos_encoded.append(torch.mean( (self.next_pos_pred[oo] - self.next_pos_encoded[oo]).pow(2)) * 10)
            loss_next_vel_encoded.append(torch.mean((self.next_vel_pred[oo] - self.next_vel_encoded[oo]).pow(2)) * 10)
        # loss_vel_dir: direction
        # dir gt
        gt_for_cross_entropy = self.gt_vel[oo].div(self.gt_vel[oo].abs() + eps).add(1).div(2)

        if dir_mode:
            '''
            loss_vel_dir = torch.mean( -( self.gt_for_cross_entropy[0] * torch.log(self.dir+eps) + (1-self.gt_for_cross_entropy[0])*torch.log(1-self.dir+eps) ) )
            for o in range(1, n_obj):
                loss_vel_dir = torch.min( loss_vel_dir, torch.mean( -( self.gt_for_cross_entropy[o] * torch.log(self.dir+eps) + (1-self.gt_for_cross_entropy[o])*torch.log(1-self.dir+eps) ) ) )
            loss_vel_dir = loss_vel_dir * 10
            '''
            loss_vel_dir.append(torch.mean( -( gt_for_cross_entropy * torch.log(self.dir[oo]+eps) + (1-gt_for_cross_entropy)*torch.log(1-self.dir[oo]+eps) ) ) * 10)
            '''
            loss_vel_speed = torch.mean((self.mag - self.input_gt_vel[0] * (self.gt_for_cross_entropy[0]*2-1) ).pow(2))
            for o in range(1, n_obj):
                loss_vel_speed = torch.min( loss_vel_speed, torch.mean((self.mag - self.input_gt_vel[o] * (self.gt_for_cross_entropy[o]*2-1) ).pow(2)))
            loss_vel_speed *= 10
            '''
            loss_dir_entropy.append(10 * torch.mean( (self.dir[oo]) * (1-self.dir[oo]) ))
            loss_vel_speed.append(torch.mean((self.mag[oo] - self.gt_vel[oo] * (gt_for_cross_entropy*2-1) ).pow(2)) * 10)
        else:
            loss_vel_dir = -1
            loss_vel_speed = -1
        # inverse loss
        #loss_inverse = torch.mean((self.vel_inverse - self.gt_inverse).pow(2)) * 10


        # vel prediction loss
        if(pred_mode):

            loss_pred_last_vel = -1
            if type(self.next_gt_vel) != type(None):
                loss_pred_vel.append(torch.mean((self.next_gt_vel_real[oo] - self.next_vel_pred[oo]).pow(2)) * 10)
                if (not self.opts.test_mode) and last_vel_loss_mode:
                    #loss_pred_last_vel = torch.mean((self.last_gt_vel - self.last_vel_pred).pow(2)) * 10
                    loss_pred_last_vel = 0

        loss_sharp_whole = 0
        if (sharp_whole_mode and train):
            loss_sharp_whole = torch.mean((self.images_s_whole - self.sharp_pred_whole).pow(2)) * 100
        loss_sharp = 0
        if (sharp_loss_mode and (not sharp_whole_mode) and train) or (sharp_gen_mode and train and (not sharp_whole_mode)):
            loss_sharp = torch.mean((self.images_a - self.sample_sharp_gen).pow(2)) * 100

    if(gen_mode):
        # for test, ssim loss
        if not train:
            #self.ssim_err = self.ssim_loss(self.post_B_pred*0.5+0.5, self.post_B*0.5+0.5)
            self.ssim_err = -1
        # loss_gen: generator loss
        #loss_gen = torch.mean((self.fake_B_encoded- self.input_B).pow(2)) * 100

        if(vel_secret_mode):
            loss_next_gen = torch.mean((self.post_B_pred_recons - self.post_B).pow(2)) * 100
            self.loss_next_gen = loss_next_gen

        # content loss
        loss_content = -1

        # gt로 블러를 줬을 때 올바르게 생성하는지
        loss_gen_gt_vel = torch.mean((self.recons_B_with_gt - self.input_B).pow(2)) * 100
        loss_gen = torch.mean((self.recons_B - self.input_B).pow(2)) * 100

    if(pred_mode and gen_mode and vel_secret_mode):
        # vel GT가 사라졌으므로, vel을 훈련시킬 수 있도록 하는 가이드가 필요.
        # loss_next_pos_2: vel이 적당한 값을 가질 수 있게 함
        # loss_inverse_2: vel이 정확한 정보를 캐치할 수 있게 도와줌
        # loss_gen: vel이 적당한 값을 가질 수 있게 함
        # loss_inverse_dir: vel_dir가 정확한 정보를 캐치할 수 있게 도와줌
        # pos는 있다. pos는 loss_pos로 하면 된다.
        # 그렇다면 pos가 없다면?
        # 그냥 loss_gen으로 가능?
        #loss_G =  loss_next_pos_2 + loss_inverse_2 + loss_gen + loss_pos + loss_inverse_dir
        #loss_G = (loss_next_gen + loss_gen) #* 0.1
        msssimm_loss = pytorch_msssim.msssim(self.recons_B, self.input_B)
        msssimm_pred_loss = pytorch_msssim.msssim(self.post_B_pred_recons, self.post_B)
        #loss_G = msssimm_loss + msssimm_pred_loss + loss_next_gen + loss_gen
        loss_G = loss_next_gen #+ msssimm_pred_loss
        #loss_G = 0
        '''
        for o in range(n_obj):
            loss_G += loss_next_pos_encoded[o]
            loss_G += loss_next_vel_encoded[o]
        '''
    elif(pred_mode and gen_mode and encoder_tuning_mode):
        msssimm_loss = pytorch_msssim.msssim(self.recons_B, self.input_B)
        loss_G = (loss_gen + msssimm_loss)
        #loss_G *= 0.1
    elif(pred_mode and gen_mode):
        #loss_G = loss_vel_speed + loss_pred_vel + loss_pred_last_vel + loss_vel_dir + loss_vel + loss_gen_gt_vel + loss_content + loss_pos + loss_sharp
        # speed loss
        #loss_G = loss_vel_speed + loss_pred_vel + loss_vel_dir + loss_gen_gt_vel + loss_pos + loss_vel# + loss_pred_last_vel + loss_content + loss_sharp + loss_next_pos_pred# + loss_gen + loss_inverse + loss_kl_za_b + percp_loss_I + loss_gt
        if throwing_mode and (type(self.given_vel) != type(None)):
            loss_G = 0
            for o in range(n_obj):
                loss_G += (loss_pred_vel[o])
        else:
            loss_G = loss_gen_gt_vel + 0
            for o in range(n_obj):
                loss_G += (loss_vel_dir[o] + loss_vel[o] + loss_pos[o] + loss_vel_speed[o] + loss_dir_entropy[o] + loss_pred_vel[o])
    elif(pred_mode):
        loss_G = loss_pred_vel + loss_pred_last_vel + loss_vel_dir + loss_vel + loss_pos
    elif(gen_mode):
        loss_G = loss_gen_gt_vel + 0
        for o in range(n_obj):
            loss_G += (loss_vel_dir[o] + loss_vel[o] + loss_pos[o] + loss_vel_speed[o] + loss_dir_entropy[o])
    else:
        loss_G = 0
        for o in range(n_obj):
            loss_G += loss_vel_dir[o] + loss_vel[o] + loss_pos[o] + loss_vel_speed[o] + loss_dir_entropy[o]
    if(train):
        loss_G.backward(retain_graph=True)

    self.loss_inverse_dir = -1
    self.loss_inverse_2 = -1
    self.loss_next_pos_2 = loss_next_pos_2
    #self.loss_inverse = loss_inverse.item()
    self.loss_inverse = -1


    self.loss_content = -1
    self.loss_pred_vel = -1
    self.loss_pred_last_vel = -1
    self.loss_sharp_whole = -1
    self.loss_sharp = loss_sharp
    if gen_mode:
        self.loss_gen = loss_gen.item()
        self.loss_gen_gt_vel = loss_gen_gt_vel.item()
    else:
        self.loss_gen = -1
        self.loss_gen_gt_vel = -1
    self.loss_vel_dir = []
    self.loss_vel = []
    self.loss_pos = []
    self.loss_vel_speed = []
    self.loss_dir_entropy = []
    self.loss_pred_vel = []

    for o in range(n_obj):
        self.loss_vel.append(loss_vel[o].item())
        self.loss_pos.append(loss_pos[o].item())
        self.loss_vel_speed.append(loss_vel_speed[o].item())
        if dir_mode:
            self.loss_vel_dir.append(loss_vel_dir[o].item())
            self.loss_dir_entropy.append(loss_dir_entropy[o].item())
        else:
            self.loss_vel_dir.append(-1)
            self.loss_dir_entropy.append(-1)
        if(pred_mode):
            self.loss_pred_vel.append(loss_pred_vel[o].item())
            #self.loss_pred_last_vel = loss_pred_last_vel
        if(sharp_whole_mode and train):
            self.loss_sharp_whole = loss_sharp_whole


    self.G_loss = loss_G.item()

  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = torch.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self, train=True):
    if(gen_mode and dir_mode):
        if(vel_secret_mode):
            #loss_vel_recons = torch.mean((self.vel_recons - self.vel_pred).pow(2)) * 10
            loss_vel_recons = -1
        else:
            #loss_vel_recons = torch.mean((self.vel_recons - self.input_gt_vel).pow(2)) * 10
            loss_vel_recons = -1

        loss_G2 = loss_vel_recons
        if train:
            loss_G2.backward()
        self.loss_vel_recons = loss_vel_recons
    else:
        self.loss_vel_recons = -1

  def update_lr(self):
    self.enc_pred_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()
    if pred_relation_mode:
        self.enc_relation_sch.step()
    if sharp_gen_mode:
        self.gen_sharp_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)

    # weight
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    if (sharp_gen_mode):
        self.gen_sharp.load_state_dict(checkpoint['gen_sharp'])
    self.enc_pred.load_state_dict(checkpoint['enc_pred'])
    if pred_relation_mode:
        self.enc_relation.load_state_dict(checkpoint['enc_relation'])

    # optimizer
    if train:
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
      if (sharp_gen_mode):
          self.gen_sharp_opt.load_state_dict(checkpoint['gen_sharp_opt'])
      self.enc_pred_opt.load_state_dict(checkpoint['enc_pred_opt'])
      if pred_relation_mode:
          self.enc_relation_opt.load_state_dict(checkpoint['enc_relation_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    if pred_relation_mode:
        state = {
             'enc_pred': self.enc_pred.state_dict(),
             'enc_relation': self.enc_relation.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'enc_pred_opt': self.enc_pred_opt.state_dict(),
             'enc_relation_opt': self.enc_relation_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    elif sharp_gen_mode:
        state = {
             'enc_pred': self.enc_pred.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'gen_sharp': self.gen_sharp.state_dict(),
             'enc_pred_opt': self.enc_pred_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'gen_sharp_opt': self.gen_sharp_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    else:
        state = {
             'enc_pred': self.enc_pred.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'enc_pred_opt': self.enc_pred_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    time.sleep(10)
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_prev = self.normalize_image(self.prev_B).detach()
    images_b = self.normalize_image(self.input_B).detach()
    images_fakes = []
    for o in range(n_obj):
        images_fakes.append(self.normalize_image(self.fake_B_encoded[o]).detach())
    images_fake_pred = self.normalize_image(self.recons_B).detach()
    images_fake_gt_blur = self.normalize_image(self.recons_B_with_gt).detach()
    if pred_mode:
        images_post_pred = self.normalize_image(self.post_B_pred_recons).detach()

    batch_size, c, h, w = images_b.size()
    #images_content_i = images_content_i.view(batch_size, 3, h//4, w//4, 1).repeat(1,1,1,1,4).view(batch_size, 3, h//4, w)
    #images_content_i = images_content_i.view(batch_size, 3, h//4, 1, w).repeat(1,1,1,4,1).view(batch_size, 3, h, w)
    #images_content_b = images_content_b.view(batch_size, 3, h//4, w//4, 1).repeat(1,1,1,1,4).view(batch_size, 3, h//4, w)
    #images_content_b = images_content_b.view(batch_size, 3, h//4, 1, w).repeat(1,1,1,4,1).view(batch_size, 3, h, w)

    #print(images_content_i.shape)
    row1 = torch.cat((images_prev[0:1, ::],self.interval, images_b[0:1, ::], self.interval, images_fake_pred[0:1, ::], self.interval, images_fake_gt_blur[0:1, ::]),3)

    row2 = torch.cat((images_prev[1:2, ::],self.interval, images_b[1:2, ::], self.interval, images_fake_pred[1:2, ::], self.interval, images_fake_gt_blur[1:2, ::]),3)

    row3 = torch.cat((images_prev[2:3, ::],self.interval, images_b[2:3, ::], self.interval, images_fake_pred[2:3, ::], self.interval, images_fake_gt_blur[2:3, ::]),3)

    for o in range(n_obj):
        row1 = torch.cat( (row1, self.interval, images_fakes[o][0:1,::]),3)
        row2 = torch.cat( (row2, self.interval, images_fakes[o][1:2,::]),3)
        row3 = torch.cat( (row3, self.interval, images_fakes[o][2:3,::]),3)

    if(pred_mode):
        row1 = torch.cat((row1, self.interval, images_post_pred[0:1,::]),3)
        row2 = torch.cat((row2, self.interval, images_post_pred[1:2,::]),3)
        row3 = torch.cat((row3, self.interval, images_post_pred[2:3,::]),3)
    return torch.cat((row1,row2,row3),2)

    # 1: sharp, 2: image_prev, 3: blur, 4: generated_blur, 5: random sharp,
    # 6: 5을 blur (3의 blur 정보로), 7: gt 로 blur, 8: 4의 blur 정보로 blur, 9: random으로 blur
    # 10: 원본블러의 컨텐트로 blur, 11: 1번의 sharp이미지 복원 (속도0)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
