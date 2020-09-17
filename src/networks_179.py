import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as py_models
import numpy
import copy
import global_options

B0minusB1Mode = global_options.B0minusB1Mode
pos_gen_mode = global_options.pos_gen_mode
vel_secret_mode = global_options.vel_secret_mode
z_delete_mode= global_options.z_delete_mode
dir_mode = global_options.dir_mode
pred2_mode = global_options.pred2_mode
####################################################################
#------------------------- next value prediction --------------------------
####################################################################
class E_pred_relation(nn.Module):
  def __init__(self, norm_layer=None, nl_layer=None, h = 64, w = 64, gpu= None):
    super(E_pred_relation, self).__init__()
    norm_layer_fc = LayerNorm_FC
    self.gpu = gpu
    init_fm = 64
    self.fc_vel = nn.Sequential(*[nn.Linear(init_fm, 128), norm_layer_fc(128), nn.ReLU(), nn.Linear(128, 128), norm_layer_fc(128), nn.ReLU(), nn.Linear(128, 128), norm_layer_fc(128), nn.ReLU()], nn.Linear(128, 32))

        
  def forward(self, vel_pred, vel_comp, pos_pred, pos_comp): 
    # pos [b,2], vel [b, 32]
    batch_size = pos_pred.size()[0]
    
    pos_pred_whole = torch.zeros([batch_size, 16, 2]).cuda(self.gpu)
    cur_pos_pred = pos_pred.clone()
    vel_pred_view = vel_pred.view(-1,16,2)
    for i in range(0,16):
        cur_pos_pred = cur_pos_pred + vel_pred_view[:,i] * 0.01
        pos_pred_whole [:,i] = cur_pos_pred
    pos_pred_whole = pos_pred_whole.view(-1,32)
    
    pos_comp_whole = torch.zeros([batch_size, 16, 2]).cuda(self.gpu)
    cur_pos_comp = pos_comp.clone()
    vel_comp_view = vel_comp.view(-1,16,2)
    for i in range(0,16):
        pos_comp_whole [:,15-i] = cur_pos_comp
        cur_pos_comp = cur_pos_comp - vel_comp_view[:,15-i] * 0.01
    pos_comp_whole = pos_comp_whole.view(-1,32)
    
    cv_concat = torch.cat([pos_pred_whole, pos_comp_whole],1)
        
    output_vel = self.fc_vel(cv_concat)
    
    return output_vel


class E_pred(nn.Module):
  def __init__(self, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None, h = 64, w = 64, gpu= None):
    super(E_pred, self).__init__()
    self.nz = output_nc
    ndf = 64
    n_blocks=4
    max_ndf = 4
    norm_layer_fc = LayerNorm_FC
    self.gpu = gpu
    
    if pred2_mode:
        init_fm = 32
    else:
        init_fm = 32 + 32 + 2
    self.fc_vel = nn.Sequential(*[nn.Linear(init_fm, 128), norm_layer_fc(128), nn.ReLU(), nn.Linear(128, 128), norm_layer_fc(128), nn.ReLU(), nn.Linear(128, 128), norm_layer_fc(128), nn.ReLU()], nn.Linear(128, 32))
    #self.fc_vel = nn.Sequential(*[nn.Linear(init_fm, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 32)]) # 181
    #self.fc_vel = nn.Sequential(*[nn.Linear(init_fm, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 32)]) # 183

        
  def forward(self, content= None, pos=None, vel=None, z=None, forward_pos = False, forward_future = False): 
    if(forward_pos):
        return self.forward_pos(content)
    if(forward_future):
        return self.forward_future(pos,vel,z)
    # content [b,2], vel [b, 32], state [b, 32]
    
    if pred2_mode:
        batch_size = pos.size()[0]
        pos_pred_whole = torch.zeros([batch_size, 16, 2]).cuda(self.gpu)
        # this is current pos version
        cur_pos = pos.clone()
        # this is next pos version
        vel = vel.view(-1,16,2)
        for i in range(0,16):
            pos_pred_whole [:,15-i] = cur_pos
            cur_pos = cur_pos - vel[:,15-i] * 0.01
        pos_pred_whole = pos_pred_whole.view(-1,32)
        cv_concat = pos_pred_whole
        
    else:
        cv_concat = torch.cat([pos, vel, z],1) # [b, 2 + 32 + 32]
    output_vel = self.fc_vel(cv_concat)
    
    output_pos = output_vel
    output_z = output_vel
    


    return content, output_vel, output_pos, output_z

  def forward_pos(self, content):
    
    return content

  def forward_future(self, pos, vel, z):
    if pred2_mode:
        batch_size = pos.size()[0]
        pos_pred_whole = torch.zeros([batch_size, 16, 2]).cuda(self.gpu)
        # this is current pos version
        cur_pos = pos
        # this is next pos version
        vel = vel.view(-1,16,2)
        for i in range(0,16):
            pos_pred_whole [:,15-i] = cur_pos
            cur_pos = cur_pos - vel[:,15-i] * 0.01
        pos_pred_whole = pos_pred_whole.view(-1,32)
        cv_concat = pos_pred_whole
    else:
        cv_concat = torch.cat([pos, vel, z],1) # [b, 2 + 32 + 32]
        #cv_concat = torch.cat([pos, vel],1) # [b, 2 + 32 + 32]
    
    output_vel = self.fc_vel(cv_concat)
    
    output_pos = output_vel
    output_z = output_vel
    
    return output_vel, output_pos, output_z


####################################################################
#---------------------------- Encoders -----------------------------
####################################################################
class E_content(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, norm_layer=None, nl_layer=None):
    super(E_content, self).__init__()
    
    ndf = 32 #32
    n_blocks=  4
    stride = 2
    max_ndf = 4
    output_nc = 2
    
    # content encoding
    conv_layers_B = [nn.ReflectionPad2d(1)]
    
    if(B0minusB1Mode):
        conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=3, stride=stride, padding=0, bias=True)]
    else:
        conv_layers_B += [nn.Conv2d(input_dim_b*2, ndf, kernel_size=3, stride=stride, padding=0, bias=True)]
        
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    
    self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_ndf//2), nn.ReLU(), nn.Linear(output_ndf//2, output_ndf//4), nn.ReLU(), nn.Linear(output_ndf//4, output_nc)])
    #self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_ndf), nn.ReLU(), nn.Linear(output_ndf, output_ndf), nn.ReLU(), nn.Linear(output_ndf, output_nc)])
    self.conv_B = nn.Sequential(*conv_layers_B)

  def forward(self, xprev, xb, forward_b=None):
    if(B0minusB1Mode):
        x_cat = xb-xprev
    else:
        x_cat = torch.cat([xprev,  xb], 1)

    # pos
    x_conv_B = self.conv_B(x_cat)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B) 

    return output_B

class E_attr_concat(nn.Module):
  def __init__(self, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
    super(E_attr_concat, self).__init__()

    ndf = 32 #32
    n_blocks= 4
    stride = 2
    max_ndf = 4
    output_dir = output_nc
    # dir encoding
    dir_conv_layers = [nn.ReflectionPad2d(1)]
    
    if(B0minusB1Mode):
        dir_conv_layers += [nn.Conv2d(input_dim_b, ndf, kernel_size=3, stride=stride, padding=0, bias=True)]
    else:
        dir_conv_layers += [nn.Conv2d(input_dim_b*2, ndf, kernel_size=3, stride=stride, padding=0, bias=True)]
        
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      dir_conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    dir_conv_layers += [nl_layer(), nn.AdaptiveAvgPool2d(1)]
    self.fc_dir = nn.Sequential(*[nn.Linear(output_ndf, output_ndf), nn.ReLU(),  nn.Linear(output_ndf, output_ndf), nn.ReLU(), nn.Linear(output_ndf, output_nc), nn.Sigmoid()]) #[0,1]
    self.conv_dir = nn.Sequential(*dir_conv_layers)
    
    # content encoding
    conv_layers_B = [nn.ReflectionPad2d(1)]
    
    if(B0minusB1Mode):
        conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=3, stride=stride, padding=0, bias=True)]
    else:
        conv_layers_B += [nn.Conv2d(input_dim_b*2, ndf, kernel_size=3, stride=stride, padding=0, bias=True)]
        
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    
    #self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_ndf), nn.ReLU(),  nn.Linear(output_ndf, output_ndf), nn.ReLU(), nn.Linear(output_ndf, output_nc), nn.ReLU()])
    self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_ndf), nn.ReLU(),  nn.Linear(output_ndf, output_ndf), nn.ReLU(), nn.Linear(output_ndf, output_nc), nn.ReLU()])
    self.conv_B = nn.Sequential(*conv_layers_B)

  def forward(self, xprev, xb):
    
    if(B0minusB1Mode):
        x_cat = xb-xprev
    else:
        x_cat = torch.cat([xprev,  xb], 1)
    # dir
    if dir_mode:
        x_dir = self.conv_dir(x_cat)
        x_dir_flat = x_dir.view(x_cat.size(0), -1)
        output_dir = self.fc_dir(x_dir_flat)#.mul(2).sub(1)
    #output_dir = output_dir.view([x_cat.size(0), -1, 1, 1])
    #output_dir = output_dir.repeat(1,1,xb.size(2), xb.size(3))

    # vel
    x_conv_B = self.conv_B(x_cat)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B) 
    #outputVar_B = self.fcVar_B(conv_flat_B)
    
    # shutter speed
    #return  output_B, outputVar_B, output_dir # output_dir: [-1,1]
    if dir_mode:
        return  output_B, output_dir # output_dir: [-1,1]
    return output_B


####################################################################
#--------------------------- Generators ----------------------------
####################################################################
class G_concat(nn.Module):
  def __init__(self, output_dim_a, nz, h = 128, w = 128, gpu = None):
    super(G_concat, self).__init__()
    self.nz = nz
    self.h = h//2
    self.w = w//2
    tch = 256
    norm_layer_fc = LayerNorm_FC
    dec_share = []
    dec_fc = []
    self.gpu = gpu
    
    tch = (self.h//2) * (self.w//2)
    dec_fc += [nn.Linear(32, 32), norm_layer_fc(32), nn.ReLU(), nn.Linear(32, 128), norm_layer_fc(128), nn.ReLU(), nn.Linear(128, tch), norm_layer_fc(tch), nn.ReLU()]
    dec_share += [ReLUINSConv2d(1, 256 , kernel_size=1, stride=1, padding=0)]
    #dec_share += [INSResBlock(tch, tch)]
    tch = 256+1
    decB1 = []
    for i in range(0, 3):
      decB1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decB2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decB3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decB4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]

        
    self.dec_fc = nn.Sequential(*dec_fc)
    self.dec_share = nn.Sequential(*dec_share)
    self.decB1 = nn.Sequential(*decB1)
    self.decB2 = nn.Sequential(*[decB2])
    self.decB3 = nn.Sequential(*[decB3])
    self.decB4 = nn.Sequential(*decB4)

  def forward(self, x, z, shutter_speed):
    batch_size = x.size(0)
    pos_pred_whole = torch.zeros([batch_size, 16, 2]).cuda(self.gpu)
    # this is current pos version
    cur_pos = x
    # this is next pos version
    vel = z.view(-1,16,2)
    if pred2_mode:
        for i in range(0,16):
            pos_pred_whole [:,15-i] = cur_pos
            cur_pos = cur_pos - vel[:,15-i] * 0.01
    else:
        # 이거 뭔가 잘못된 것 같음
        for i in range(0,16):
            cur_pos = cur_pos + vel[:,i] * 0.01
            pos_pred_whole [:,i] = cur_pos
    pos_pred_whole = pos_pred_whole.view(-1,32)
    z = pos_pred_whole
    out0 = self.dec_fc(pos_pred_whole)
    out0 = out0.view(out0.size(0), 1, self.h//2, self.w//2)
    out0 = self.dec_share(out0)
    shutter_speed = shutter_speed.view(shutter_speed.size(0), shutter_speed.size(1), 1, 1).expand(shutter_speed.size(0), shutter_speed.size(1), self.h//2, self.w//2)
    x_and_z = torch.cat([out0, shutter_speed], 1)
    
    out1 = self.decB1(x_and_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.decB2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.decB3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.decB4(x_and_z4)
    
    return out4

class G_sharp(nn.Module):
  def __init__(self, output_dim_a, nz, h = 128, w = 128):
    super(G_sharp, self).__init__()
    self.nz = nz
    self.h = h//2
    self.w = w//2
    tch = 256
    norm_layer_fc = LayerNorm_FC
    dec_share = []
    dec_fc = []
    if(pos_gen_mode):
        tch = (self.h//2) * (self.w//2)
        dec_fc += [nn.Linear(2, 32), norm_layer_fc(32), nn.ReLU(), nn.Linear(32, 128), norm_layer_fc(128), nn.ReLU(), nn.Linear(128, tch), norm_layer_fc(tch), nn.ReLU()]
        dec_share += [ReLUINSConv2d(1, 256 , kernel_size=1, stride=1, padding=0)]
        #dec_share += [INSResBlock(tch, tch)]
        tch = 256
        decB1 = []
        for i in range(0, 3):
          decB1 += [INSResBlock(tch, tch)]
        tch = tch
        decB2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch//2
        tch = tch
        decB3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch//2
        tch = tch
        decB4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    else:
        dec_share += [INSResBlock(tch, tch)]
        tch = tch+self.nz+1
        decB1 = []
        for i in range(0, 3):
          decB1 += [INSResBlock(tch, tch)]
        tch = tch + self.nz
        decB2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch//2
        tch = tch + self.nz
        decB3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch//2
        tch = tch + self.nz
        decB4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
        
    self.dec_fc = nn.Sequential(*dec_fc)
    self.dec_share = nn.Sequential(*dec_share)
    self.decB1 = nn.Sequential(*decB1)
    self.decB2 = nn.Sequential(*[decB2])
    self.decB3 = nn.Sequential(*[decB3])
    self.decB4 = nn.Sequential(*decB4)

  def forward(self, x):
    if(pos_gen_mode):
        #out0 = x.view(x.size(0), x.size(1), 1, 1).expand(x.size(0), x.size(1), self.h//2, self.w//2) # [b, 2]
        out0 = self.dec_fc(x)
        #out0 = out0.view(out0.size(0), out0.size(1), 1, 1).expand(out0.size(0), out0.size(1), self.h//2, self.w//2)
        out0 = out0.view(out0.size(0), 1, self.h//2, self.w//2)
        out0 = self.dec_share(out0)
        x_and_z = out0
        #x_and_z = self.dec_share(x_and_z)
    else:
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), self.h//2, self.w//2)
        shutter_speed = shutter_speed.view(shutter_speed.size(0), shutter_speed.size(1), 1, 1).expand(shutter_speed.size(0), shutter_speed.size(1), self.h//2, self.w//2)
        x_and_z = torch.cat([out0,  z_img, shutter_speed], 1)
        
    out1 = self.decB1(x_and_z)
    out2 = self.decB2(out1)
    out3 = self.decB3(out2)
    out4 = self.decB4(out3)
    return out4

class G_manual():
  def __init__(self, sharp_gen):
    #super(G_manual, self).__init__()
    self.sharp_gen = sharp_gen

  def g(self, arr, gamma = 2.2, inverse=False):
    if(inverse):
        return arr.pow(gamma)
    else:
        return arr.pow(1/gamma)
    
  def forward(self, x, z, shutter_speed):
    batch_size = x.size(0)
    vel = z.view(-1,16,2)
    ret = torch.zeros([batch_size,self.ratio, 3,self.h,self.w]).cuda(self.gpu)
    # x: pos
    pos_x = (x[:,0] + 1) * self.w/2
    pos_y = (x[:,1] + 1) * self.h/2
    
    for i in range(self.ratio):
        ret[:, self.ratio-1-i] = self.draw(pos_x,pos_y, batch_size)
        pos_x = pos_x - vel[:,self.ratio-1-i,0] * 0.01
        pos_y = pos_y - vel[:,self.ratio-1-i,1] * 0.01
    return self.make_blur(ret, batch_size, shutter_speed)

class G_manual2():
  def __init__(self, nz, gpu, h = 128, w = 128):
    #super(G_manual, self).__init__()
    self.nz = nz
    self.ratio = nz//2
    self.h = h
    self.w = w
    self.scale = 1
    self.scaled_h = self.h * self.scale
    self.scaled_w = self.w * self.scale
    #self.size = torch.tensor(10.)
    #self.size = torch.nn.Parameter(self.size)
    self.size = 10
    self.gpu = gpu
    
  def draw(self, pos_x, pos_y, batch_size):
    ret = torch.zeros([batch_size,3,self.scaled_h, self.scaled_w]).cuda(self.gpu)
    pos_x = (pos_x * self.scale)#.int()
    pos_y = (pos_y * self.scale)#.int()
    pos_x_int = pos_x.int()
    pos_y_int = pos_y.int()
    
    radius = (self.size * self.scale)#.int()
    for b in range(batch_size):
        for x in range(pos_x_int[b]-radius, pos_x_int[b]+radius):
            for y in range(pos_y_int[b]-radius, pos_y_int[b]+radius):
                k = (x-pos_x[b]).pow(2) + (y-pos_y[b]).pow(2) - radius**2
                if ( k>0 and x >= 0 and y >=0 and x < self.scaled_w and y <self.scaled_h):
                    ret[b,:,y,x] = k
    # resize 필요
    return ret

  def g(self, arr, gamma = 2.2, inverse=False):
    if(inverse):
        return arr.pow(gamma)
    else:
        return arr.pow(1/gamma)
    
  def make_blur(self, ret, batch_size, shutter_speed):
    ret = self.g(ret, inverse=True)
    ret_final = torch.zeros([batch_size, 3,self.h,self.w]).cuda(self.gpu)
    
    for b in range(batch_size):
        for i in range(self.ratio):
            if((i/(self.ratio-1)) >= (0.5 - shutter_speed[b]*0.5)) and ((i/(self.ratio-1)) <= (0.5 + shutter_speed[b]*0.5)):
                ret_final[b] += ret[b,i]
    return self.g(ret_final)
    
  def forward(self, x, z, shutter_speed):
    batch_size = x.size(0)
    vel = z.view(-1,16,2)
    ret = torch.zeros([batch_size,self.ratio, 3,self.h,self.w]).cuda(self.gpu)
    # x: pos
    pos_x = (x[:,0] + 1) * self.w/2
    pos_y = (x[:,1] + 1) * self.h/2
    
    for i in range(self.ratio):
        ret[:, self.ratio-1-i] = self.draw(pos_x,pos_y, batch_size)
        pos_x = pos_x - vel[:,self.ratio-1-i,0] * 0.01
        pos_y = pos_y - vel[:,self.ratio-1-i,1] * 0.01
    return self.make_blur(ret, batch_size, shutter_speed)

####################################################################
#--------------------------- losses ----------------------------
####################################################################
class PerceptualLoss():
    def __init__(self, loss, gpu="cuda", p_layer=14):
        super(PerceptualLoss, self).__init__()
        self.criterion = loss
        self.gpu = gpu
        
        cnn = py_models.vgg19(pretrained=True).features
        cnn = cnn.cuda(self.gpu)
        model = nn.Sequential()
        model = model.cuda(self.gpu)
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == p_layer:
                break
        self.contentFunc = model     

    def getloss(self, fakeIm, realIm):
        if isinstance(fakeIm, numpy.ndarray):
            fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda(self.gpu)
            realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda(self.gpu)
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
class PerceptualLoss16():
    def __init__(self, loss, gpu="cuda", p_layer=14):
        super(PerceptualLoss16, self).__init__()
        self.criterion = loss
        self.gpu = gpu
#         conv_3_3_layer = 14
        checkpoint = torch.load('/vggface_path/VGGFace16.pth')
        vgg16 = py_models.vgg16(num_classes=2622)
        vgg16.load_state_dict(checkpoint['state_dict'])
        cnn = vgg16.features
        cnn = cnn.cuda(gpu)
#         cnn = cnn.to(gpu)
        model = nn.Sequential()
        model = model.cuda(gpu)
        for i,layer in enumerate(list(cnn)):
#             print(layer)
            model.add_module(str(i),layer)
            if i == p_layer:
                break
        self.contentFunc = model   
        del vgg16, cnn, checkpoint

    def getloss(self, fakeIm, realIm):
        if isinstance(fakeIm, numpy.ndarray):
            fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda(self.gpu)
            realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda(self.gpu)
        
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    
class GradientLoss():
    def __init__(self, loss, n_scale=3):
        super(GradientLoss, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.criterion = loss    
        self.n_scale = n_scale
        
    def grad_xy(self, img):
        gradient_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        gradient_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gradient_x, gradient_y

    def getloss(self, fakeIm, realIm):
        loss = 0
        for i in range(self.n_scale):
            fakeIm = self.downsample(fakeIm)
            realIm = self.downsample(realIm)
            grad_fx, grad_fy = self.grad_xy(fakeIm)
            grad_rx, grad_ry = self.grad_xy(realIm)            
            loss += pow(4,i) * self.criterion(grad_fx, grad_rx) + self.criterion(grad_fy, grad_ry)
        return loss  

class l1GradientLoss():
    def __init__(self, loss, n_scale=3):
        super(l1GradientLoss, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.criterion = loss    
        self.n_scale = n_scale
        
    def grad_xy(self, img):
        gradient_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        gradient_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gradient_x, gradient_y

    def getloss(self, fakeIm):
        loss = 0
        for i in range(self.n_scale):
            fakeIm = self.downsample(fakeIm)
            grad_fx, grad_fy = self.grad_xy(fakeIm)       
            loss += self.criterion(grad_fx, torch.zeros_like(grad_fx)) + self.criterion(grad_fy, torch.zeros_like(grad_fy))
        
        return loss   

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += conv3x3(inplanes, outplanes)
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
def conv3x3(in_planes, out_planes):
  return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class LayerNorm_FC(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm_FC, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out))
      self.bias = nn.Parameter(torch.zeros(n_out))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight, self.bias)
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

