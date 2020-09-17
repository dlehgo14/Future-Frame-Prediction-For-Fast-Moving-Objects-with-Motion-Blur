import torch
from options import TrainOptions
from dataset import *
from model2 import UID
from saver import Saver
import numpy as np
from PIL import Image
from skimage.draw import circle
from skimage.transform import resize
import os
import torch.multiprocessing as multiprocessing
import global_options

vel_secret_mode = global_options.vel_secret_mode
sharp_whole_mode = global_options.sharp_whole_mode
encoder_tuning_mode = global_options.encoder_tuning_mode
z_delete_mode = global_options.z_delete_mode
n_obj = global_options.n_obj
pred_mode = global_options.pred_mode
throwing_mode = global_options.throwing_mode



data_version = "1ball_NoCollision" + ".npz"


test_ratio = 20
if vel_secret_mode or encoder_tuning_mode:
    test_ratio = 4

def make_img(base, coord, circle_radius, color, scale, size):
    rr, cc = circle(int((coord[1])*scale), int((coord[0])*scale), circle_radius*scale, [size*scale, size*scale])
    base[rr,cc] = color
    return base

def save_pos_for_next_gt(save_folder, save_name,isVel=True, gt_next_=None, pred_next_=None):
    try:
        if not(os.path.isdir(save_folder)):
            os.makedirs(os.path.join(save_folder))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    size = 128
    scale = 10
    circle_radius = 3
    save_path = save_folder + "/"+save_name

    green = np.array([0, 255, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype = np.uint8)
    yellow = np.array([255, 255, 0], dtype = np.uint8)

    interval = np.ones([10, size , 3], dtype=np.uint8) * 255
    ret = np.ones([10, size , 3], dtype=np.uint8) * 255
    for i in range(gt_next_.size()[0]):
        gt_next = gt_next_[i].cpu().detach().numpy()
        pred_next = pred_next_[i].cpu().detach().numpy()
        gt_next = np.reshape(gt_next, [-1,2])
        pred_next = np.reshape(pred_next, [-1,2])
        gt_next = (gt_next)*size*0.1
        pred_next = (pred_next)*size*0.1
        # 4
        base4 = np.zeros([size * scale, size * scale, 3], dtype=np.float)
        gt_base = np.zeros([2],dtype=np.float) + size*0.5
        pred_base = np.zeros([2],dtype=np.float) + size*0.5
        for s in range(gt_next.shape[0]):
            gt_base += gt_next[s]/5
            pred_base += pred_next[s]/5
            base4 = make_img(base4, (gt_base), circle_radius, blue, scale, size)
            base4 = make_img(base4, (pred_base), circle_radius, yellow, scale, size)
        base4 = resize(base4, [size, size], anti_aliasing=True).astype(np.uint8)
        ret = np.concatenate([ret, base4, interval], axis=0)
    #print(base.shape)
    img = Image.fromarray(ret, 'RGB')
    img.save(save_path)

def save_pos(gt_, pred_, pred2_,gt_inverse_, pred_inverse_, save_folder, save_name, isVel=True, gt_next_=None, pred_next_=None, gt_pos_=None, pred_pos_=None, gt_pos_2=None):
    try:
        if not(os.path.isdir(save_folder)):
            os.makedirs(os.path.join(save_folder))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    size = 256
    scale = 10
    circle_radius = 2
    save_path = save_folder + "/"+save_name

    green = np.array([0, 255, 0], dtype=np.uint8)
    blue = np.array([255, 0, 255], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype = np.uint8)
    yellow = np.array([255, 255, 0], dtype = np.uint8)



    interval = np.ones([10, size , 3], dtype=np.uint8) * 255
    ret = np.ones([10, size , 3], dtype=np.uint8) * 255
    for i in range(gt_.size()[0]):
        gt = gt_[i].cpu().detach().numpy()
        pred = pred_[i].cpu().detach().numpy()
        pred2 = pred2_[i].cpu().detach().numpy()
        gt_inverse = gt_inverse_[i].cpu().detach().numpy()
        pred_inverse = pred_inverse_[i].cpu().detach().numpy()

        gt = np.reshape(gt, [-1,2])
        pred = np.reshape(pred, [-1,2])
        pred2 = np.reshape(pred2, [-1,2])
        gt_inverse = np.reshape(gt_inverse, [-1,2])
        pred_inverse = np.reshape(pred_inverse, [-1,2])

        if (type(gt_next_) != type(None)):
            gt_next = gt_next_[i].cpu().detach().numpy()
            pred_next = pred_next_[i].cpu().detach().numpy()
            gt_next = np.reshape(gt_next, [-1,2])
            pred_next = np.reshape(pred_next, [-1,2])
            gt_next = (gt_next)*size*0.1
            pred_next = (pred_next)*size*0.1

        if (type(gt_pos_) != type(None)):
            gt_pos = gt_pos_[i].cpu().detach().numpy()
            pred_pos = pred_pos_[i].cpu().detach().numpy()
            gt_pos = np.reshape(gt_pos, [-1,2])
            pred_pos = np.reshape(pred_pos, [-1,2])
            gt_pos = (gt_pos+1)*size*0.5
            pred_pos = (pred_pos+1)*size*0.5

        if (type(gt_pos_2) != type(None)):
            gt_pos2 = gt_pos_2[i].cpu().detach().numpy()
            gt_pos2 = np.reshape(gt_pos2, [-1,2])
            gt_pos2 = (gt_pos2+1)*size*0.5
        if(isVel):
            gt = (gt)*size*0.1
            pred = (pred)*size*0.1
            pred2 = (pred2)*size*0.1
            gt_inverse = (gt_inverse)*size*0.1
            pred_inverse = (pred_inverse)*size*0.1
        else:
            gt = (gt+1)*size*0.5
            pred = (pred+1)*size*0.5
            pred2 = (pred2+1)*size*0.5
        length = gt.shape[0]
        # 1
        base = np.zeros([size * scale, size * scale, 3], dtype=np.float)
        gt_base = np.zeros([2],dtype=np.float) + size*0.5
        pred_base = np.zeros([2],dtype=np.float) + size*0.5
        for s in range(length):
            if(isVel):
                gt_base += gt[s]/20
                pred_base += pred[s]/20
                #print(gt[s])
                base = make_img(base, (gt_base), circle_radius, blue, scale, size)
                base = make_img(base, (pred_base), circle_radius, green, scale, size)
            else:
                base = make_img(base, (gt[s]), circle_radius, blue, scale, size)
                base = make_img(base, (pred[s]), circle_radius, green, scale, size)
        base = resize(base, [size, size], anti_aliasing=True).astype(np.uint8)
        # 2
        base2 = np.zeros([size * scale, size * scale, 3], dtype=np.float)
        gt_base = np.zeros([2],dtype=np.float) + size*0.5
        pred_base = np.zeros([2],dtype=np.float) + size*0.5
        for s in range(length):
            if(isVel):
                gt_base += gt[s]/20
                pred_base += pred2[s]/20
                base2 = make_img(base2, (gt_base), circle_radius, blue, scale, size)
                base2 = make_img(base2, (pred_base), circle_radius, red, scale, size)
            else:
                base2 = make_img(base2, (gt[s]), circle_radius, blue, scale, size)
                base2 = make_img(base2, (pred2[s]), circle_radius, red, scale, size)
        base2 = resize(base2, [size, size], anti_aliasing=True).astype(np.uint8)
        # 3
        if(isVel):
            base3 = np.zeros([size * scale, size * scale, 3], dtype=np.float)
            gt_base = np.zeros([2],dtype=np.float) + size*0.5
            pred_base = np.zeros([2],dtype=np.float) + size*0.5
            gt_base2 = np.zeros([2],dtype=np.float) + size*0.5
            for s in range(length):
                gt_base += gt_inverse[s]/20
                pred_base += pred_inverse[s]/20
                gt_base2 += gt[s]/20
                base3 = make_img(base3, (gt_base), circle_radius, blue, scale, size)
                base3 = make_img(base3, (pred_base), circle_radius, white, scale, size)
                base3 = make_img(base3, (gt_base2), circle_radius, blue, scale, size)
            base3 = resize(base3, [size, size], anti_aliasing=True).astype(np.uint8)
            ret = np.concatenate([ret, base,interval, base2, interval, base3, interval], axis=0)
            # 4
            if( type(gt_next_) != type(None)):
                base4 = np.zeros([size * scale, size * scale, 3], dtype=np.float)
                gt_base = np.zeros([2],dtype=np.float) + size*0.5
                pred_base = np.zeros([2],dtype=np.float) + size*0.5
                for s in range(gt_next.shape[0]):
                    gt_base += gt_next[s]/20
                    pred_base += pred_next[s]/20
                    base4 = make_img(base4, (gt_base), circle_radius, blue, scale, size)
                    base4 = make_img(base4, (pred_base), circle_radius, yellow, scale, size)
                base4 = resize(base4, [size, size], anti_aliasing=True).astype(np.uint8)
                ret = np.concatenate([ret, base4, interval], axis=0)
            # 5
            if( type(gt_pos_) != type(None)):
                base5 = np.zeros([size * scale, size * scale, 3], dtype=np.float)
                for s in range(gt_pos.shape[0]):
                    base5 = make_img(base5, (gt_pos[s]), circle_radius, blue, scale, size)
                    base5 = make_img(base5, (pred_pos[s]), circle_radius, red, scale, size)
                    if( type(gt_pos_2) != type(None) ):
                        base5 = make_img(base5, (gt_pos2[s]), circle_radius, blue, scale, size)
                base5 = resize(base5, [size, size], anti_aliasing=True).astype(np.uint8)
                ret = np.concatenate([ret, base5, interval, interval], axis=0)
        else:
            ret = np.concatenate([ret, base,interval, base2, interval], axis=0)
    #print(base.shape)
    img = Image.fromarray(ret, 'RGB')
    img.save(save_path)

def main():
    # for multi processing
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn')
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    if(z_delete_mode):
        z_zero_train = torch.zeros(opts.batch_size, 32).cuda(opts.gpu)
        z_zero_test = torch.zeros(opts.batch_size_test, 32).cuda(opts.gpu)

    # daita loader
    print('\n--- load dataset ---')
    sharp_whole_root = None
    if(sharp_whole_mode):
        sharp_whole_root = "../datasets_npz/datasets_sharp_whole_"+data_version
    dataset = dataset_pair_group_simple(opts,"../datasets_npz/datasets_blur_"+data_version,"../datasets_npz/datasets_sharp_"+data_version, "../datasets_npz/datasets_sharp_start_"+data_version, "../datasets_npz/datasets_gt_vel_"+data_version, "../datasets_npz/datasets_gt_pos_"+data_version, "../datasets_npz/datasets_shutter_speed_"+data_version, sharp_whole_root= sharp_whole_root)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    dataset_test = dataset_pair_group_simple(opts,"../datasets_npz/datasets_blur_"+data_version,"../datasets_npz/datasets_sharp_"+data_version, "../datasets_npz/datasets_sharp_start_"+data_version, "../datasets_npz/datasets_gt_vel_"+data_version, "../datasets_npz/datasets_gt_pos_"+data_version, "../datasets_npz/datasets_shutter_speed_"+data_version, test_mode = True, sharp_whole_root=sharp_whole_root)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size_test, shuffle=False, num_workers=opts.nThreads)
    test_iter = iter(test_loader)

    # model
    print('\n--- load model ---')
    model = UID(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        for it, (images_b_s, gt_s, gt_pos_s, shutter_speed_s, images_s_whole_s, images_a_s) in enumerate(train_loader):
            if images_b_s.size(0) != opts.batch_size or images_b_s.size(0) != opts.batch_size:
                continue
            images_b_s = images_b_s.cuda(opts.gpu).detach()
            for o in range(n_obj):
                gt_s[o] = gt_s[o].cuda(opts.gpu).detach()
                gt_pos_s[o] = gt_pos_s[o].cuda(opts.gpu).detach()
            shutter_speed_s = shutter_speed_s.cuda(opts.gpu).detach()
            if(sharp_whole_mode):
                images_s_whole_s = images_s_whole_s.cuda(opts.gpu).detach()
            else:
                images_a_s = images_a_s.cuda(opts.gpu).detach()

            # update model
            s = 0
            z = None
            #last_gt = gt_s[:,opts.sequence_num - 1,:]
            last_gt = None
            loss_pred_last_sum = 0
            while (s < (opts.sequence_num - 2)):
                if(z_delete_mode):
                    z = z_zero_train
                # now: s+1, max: sequence_num-1
                left_steps = opts.sequence_num - s - 2 # 현재(s+1)스텝으로부터 몇번까지 더 gt 가 있는지

                images_prev = images_b_s[:,s,:,:,:]
                images_b = images_b_s[:,s+1,:,:,:]
                images_post = images_b_s[:,s+2,:,:,:]
                gt = []
                gt_pos = []
                for o in range(n_obj):
                    gt.append(gt_s[o][:,s+1,:])
                    gt_pos.append(gt_pos_s[o][:,s+1,:])
                next_gt = []
                next_gt_pos = []
                for o in range(n_obj):
                    next_gt.append(gt_s[o][:,s+2,:])
                    next_gt_pos.append(gt_pos_s[o][:,s+2,:])
                shutter_speed = shutter_speed_s
                if(sharp_whole_mode):
                    images_s_whole = images_s_whole_s[:,(s+1)*16:(s+2)*16,:,:,:]
                    images_a = None
                else:
                    images_s_whole = None
                    images_a = images_a_s[:,s+1,:,:,:]

                if throwing_mode and s != 0:
                    given_vel = model.next_vel_pred
                    given_pos = model.next_pos_pred
                else:
                    given_vel = None
                    given_pos = None

                model.update(images_prev, images_post, images_b, gt, gt_pos, next_gt = next_gt, next_gt_pos = next_gt_pos, z = z, last_gt = last_gt, left_steps=left_steps, gt_pos_set = gt_pos_s, shutter_speed=shutter_speed, images_s_whole = images_s_whole, images_a = images_a, given_vel = given_vel, given_pos = given_pos)
                s += 1
                if(model.pred_mode):
                    z = model.z_next
                    loss_pred_last_sum += model.loss_pred_last_vel
                else:
                    z = None
                    loss_pred_last_sum += -1

            # save to display file
            if (it+1) % 1 == 0:
                print('total_it: %d (ep %d, it %d), lr %08f' % (total_it+1, ep, it+1, model.enc_c_opt.param_groups[0]['lr']))
                if pred_mode:
                    loss_pred_vel = model.loss_pred_vel[0]
                else:
                    loss_pred_vel = -1
                print('gen_loss: %04f, vel_loss %04f, vel_recons_loss %04f, gen_loss_gt_vel %04f, inverse_loss %04f, vel_dir_loss %04f, loss_content %04f, vel_pred_loss %04f, pos_loss %04f, last_vel_pred_loss %04f, whole_pos_pred %04f, loss_sharp %04f' % (model.loss_gen, model.loss_vel[0], model.loss_vel_recons, model.loss_gen_gt_vel, model.loss_inverse, model.loss_vel_dir[0], model.loss_content, loss_pred_vel, model.loss_pos[0], loss_pred_last_sum/s, model.loss_sharp_whole, model.loss_sharp))
                #print(model.next_gt)
                #print(model.next_gt_pred)

            if (it+1) % test_ratio == 0:
                n = 0
                total_dir_loss = 0
                total_speed_loss = 0
                total_vel_loss = 0
                total_pos_loss = 0
                total_dir_entropy = 0
                for rep in range(1):
                    try:
                        images_b_s, gt_s, gt_pos_s, shutter_speed_s, images_s_whole_s, images_a_s = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        images_b_s, gt_s, gt_pos_s, shutter_speed_s, _, _ = next(test_iter)
                    while(images_b_s.size(0) != opts.batch_size_test):
                        try:
                            images_b_s, gt_s, gt_pos_s, shutter_speed_s, _, _ = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            images_b_s, gt_s, gt_pos_s, shutter_speed_s, _, _ = next(test_iter)

                    images_b_s = images_b_s.cuda(opts.gpu).detach()
                    for o in range(n_obj):
                        gt_s[o] = gt_s[o].cuda(opts.gpu).detach()
                        gt_pos_s[o] = gt_pos_s[o].cuda(opts.gpu).detach()
                    shutter_speed_s = shutter_speed_s.cuda(opts.gpu).detach()

                    # test model
                    s = 0
                    loss_pred_last_sum = 0
                    #last_gt = gt_s[:,opts.sequence_num - 1,:]
                    last_gt = None
                    z = None
                    while (s < (opts.sequence_num - 2)):
                        if(z_delete_mode):
                            z = z_zero_test
                        left_steps = opts.sequence_num - s - 2 # 현재(s+1)스텝으로부터 몇번까지 더 gt 가 있는지
                        images_prev = images_b_s[:,s,:,:,:]
                        images_b = images_b_s[:,s+1,:,:,:]
                        images_post = images_b_s[:,s+2,:,:,:]

                        gt = []
                        gt_pos = []
                        for o in range(n_obj):
                            gt.append(gt_s[o][:,s+1,:])
                            gt_pos.append(gt_pos_s[o][:,s+1,:])
                        next_gt = []
                        next_gt_pos = []
                        for o in range(n_obj):
                            next_gt.append(gt_s[o][:,s+2,:])
                            next_gt_pos.append(gt_pos_s[o][:,s+2,:])
                        loss_pred_last_sum += model.loss_pred_last_vel
                        shutter_speed = shutter_speed_s
                        if throwing_mode and s != 0:
                            given_vel = model.next_vel_pred
                            given_pos = model.next_pos_pred
                        else:
                            given_vel = None
                            given_pos = None
                        model.test(images_prev, images_post, images_b, gt, gt_pos, next_gt, next_gt_pos, z, last_gt = last_gt, left_steps=left_steps, gt_pos_set = gt_pos_s, shutter_speed=shutter_speed, given_vel = given_vel, given_pos = given_pos)
                        s += 1
                        total_dir_loss += model.loss_vel_dir[0]
                        total_speed_loss += model.loss_vel_speed[0]
                        total_vel_loss += model.loss_vel[0]
                        total_pos_loss += model.loss_pos[0]
                        total_dir_entropy += model.loss_dir_entropy[0]
                        n += 1

                        if(model.pred_mode):
                            z = model.z_next
                        else:
                            z = None
                #print("dir")
                #print(model.dir)
                total_dir_loss /= n
                total_speed_loss /= n
                total_vel_loss /= n
                total_pos_loss /= n
                total_dir_entropy /= n
                print('=============================')
                print("n: "+str(n))
                if(vel_secret_mode):
                    print('gen_loss: %04f, vel_loss %04f, vel_recons_loss %04f, gen_loss_gt_vel %04f, inverse_loss %04f, vel_dir_loss %04f, loss_content %04f, vel_pred_loss %04f, pos_loss %04f, dir_entropy_loss %04f, speed_loss %04f' % (model.loss_gen, total_vel_loss, model.loss_vel_recons, model.loss_gen_gt_vel, model.loss_inverse, total_dir_loss, model.loss_content, loss_pred_vel, total_pos_loss, total_dir_entropy, total_speed_loss))
                else:
                    if pred_mode:
                        loss_pred_vel = model.loss_pred_vel[0]
                    else:
                        loss_pred_vel = -1
                    print('gen_loss: %04f, vel_loss %04f, vel_recons_loss %04f, gen_loss_gt_vel %04f, inverse_loss %04f, vel_dir_loss %04f, loss_content %04f, vel_pred_loss %04f, pos_loss %04f, dir_entropy_loss %04f, speed_loss %04f' % (model.loss_gen, total_vel_loss, model.loss_vel_recons, model.loss_gen_gt_vel, model.loss_inverse, total_dir_loss, model.loss_content, loss_pred_vel, total_pos_loss, total_dir_entropy, total_speed_loss))
                print('=============================')
                if(model.gen_mode):
                    if ep%10==0 and (it+1) % 60 == 0:
                        saver.write_img(ep*len(train_loader) + (it+1), model)
                if vel_secret_mode or encoder_tuning_mode:
                    if (ep+1)%5==0:
                        save_pos(model.input_gt_vel[0], model.vel_pred[0], model.vel_pred[0], model.next_vel_pred[0], model.next_vel_encoded[0], opts.visualize_root,"example%06d.png"%(ep*len(train_loader) + (it+1)), gt_next_ = model.next_vel_encoded[0], pred_next_ = model.next_vel_pred[0], gt_pos_ = model.next_pos_encoded[0], pred_pos_ = model.next_pos_pred[0])
                elif(model.gen_mode and model.pred_mode):
                    if ep%10==0 and (it+1) % 60 == 0:
                        save_pos(model.input_gt_vel[0], model.vel_pred[0], model.vel_pred[0], model.input_gt_vel[0], model.vel_pred[0], opts.visualize_root,"example%06d.png"%(ep*len(train_loader) + (it+1)), gt_next_ = model.next_gt_vel[0], pred_next_ = model.next_vel_pred[0], gt_pos_ = model.input_gt_pos[0], pred_pos_ = model.pos_pred[0])
                elif(model.pred_mode):
                    if (it+1) % 300 == 0:
                        save_pos(model.input_gt_vel[0], model.vel_pred, model.vel_pred, model.input_gt_vel[0], model.vel_pred, opts.visualize_root,"example%06d.png"%(ep*len(train_loader) + (it+1)), gt_next_ = model.next_gt_vel, pred_next_ = model.next_vel_pred, gt_pos_ = model.input_gt_pos, pred_pos_ = model.pos_pred)
                elif(model.gen_mode):
                    if ep%10==0 and (it+1) % 60 == 0:
                        save_pos(model.input_gt_vel[0], model.vel_pred[0], model.vel_pred[1], model.input_gt_vel[0], model.vel_pred[0], opts.visualize_root,"example%06d.png"%(ep*len(train_loader) + (it+1)), gt_next_ = model.input_gt_vel[0], pred_next_ = model.vel_pred[0], gt_pos_ = model.input_gt_pos[0], pred_pos_ = model.pos_pred[0], gt_pos_2=model.input_gt_pos[0])
                else:
                    if ep%10==0 and (it+1) % 20 == 0:
                        save_pos(model.input_gt_vel[0], model.vel_pred, model.vel_pred, model.input_gt_vel[0], model.vel_pred, opts.visualize_root,"example%06d.png"%(ep*len(train_loader) + (it+1)), gt_next_ = model.input_gt_vel[0], pred_next_ = model.vel_pred, gt_pos_ = model.input_gt_pos[0], pred_pos_ = model.pos_pred, gt_pos_2=model.input_gt_pos[0])
                #save_pos_for_next_gt(opts.visualize_root,"example%06d.png"%(ep*len(train_loader) + (it+1)), gt_next_ = model.next_gt, pred_next_ = model.next_gt_pred)

            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model)
                break

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # Save network weights
        #if vel_secret_mode or encoder_tuning_mode:
            #saver.write_model(ep, total_it+1, model)
        if (ep+1)%5==0:
            saver.write_model(ep, total_it+1, model)

    return

if __name__ == '__main__':
  main()
