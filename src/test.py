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
import global_options
from skimage.measure import compare_ssim

origin_version = global_options.origin_version
pos_gen_mode = global_options.pos_gen_mode
z_delete_mode = global_options.z_delete_mode
n_obj = global_options.n_obj
throwing_mode = global_options.throwing_mode
test_split = True
convert_datasets = ["final_drift_small_3reps" + ".npz", "final_drift_test" + ".npz", "final_drift_test_2" + ".npz", "final_drift_test_2_full_shutter" + ".npz", \
"final_drift_test_2_low_shutter" + ".npz", "final_drift_test_1000" + ".npz", "final_drift_test_1000_after_400frames" + ".npz", "2balls_NoCollision" + ".npz", \
"3balls_NoCollision" + ".npz", "4balls_NoCollision" + ".npz", "5balls_NoCollision" + ".npz", "1ball_NoCollision" + ".npz"]


data_version = "1ball_NoCollision" + ".npz"

def make_img(base, coord, circle_radius, color, scale, size_w, size_h):
    # x: -1 -> 0 / 1 -> 160
    # y: 0.75 -> 0 / -0.75 -> 120
    ratio = 75
    x = int((coord[0] + 1) * ratio)
    y = int((-coord[1]+0.75) * ratio)
    #rr, cc = circle(int((-coord[1]*30 + size_h//2)*scale), int((coord[0] * 30 + 5)*scale), circle_radius*scale, [size_h*scale, size_w*scale])
    rr, cc = circle(y*scale, x*scale, circle_radius*scale, [size_h*scale, size_w*scale])
    base[rr,cc] = color
    return base

def show_trajectory(save_folder, save_dir, gt, pred):
    try:
        if not(os.path.isdir(save_folder)):
            os.makedirs(os.path.join(save_folder))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    size_w = 700
    size_h = 500
    scale = 10
    circle_radius = 2
    save_path = save_folder +save_dir

    red = np.array([255, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype = np.uint8)
    # gt [b, sequence-1, 2]
    # pred [b, sequence-1, 2]
    sequence = gt[0].shape[1]
    for b in range(1):
        base = np.zeros([size_h * scale, size_w * scale, 3], dtype=np.float)
        for o in range(0,n_obj):
            for s in range(sequence):
                base = make_img(base, gt[o][b,s], circle_radius, red, scale, size_w, size_h)
                base = make_img(base, pred[o][b,s], circle_radius, white, scale, size_w, size_h)
        base = resize(base, [size_h, size_w], anti_aliasing=True).astype(np.uint8)
    #print(base.shape)
    img = Image.fromarray(base, 'RGB')
    img.save(save_path)


def conver_to_straight_moving(pos_, vel_):
    # 다시 짠것
    ret = np.zeros([2,19,2])
    for b in range(2):
        x_dir = 1
        y_dir = 1
        x_offset = 0
        y_offset = 0
        pos = pos_[b].reshape([19,2])
        vel = vel_[b].reshape([19,16, 2])
        prev_pos = pos[0] + 0
        prev_vel = vel[0] + 0
        if prev_vel[-1][0] < 0 : x_dir = -1
        if prev_vel[-1][1] < 0 : y_dir = -1
        for i in range(1,19):
            cur_vel = vel[i] + 0
            for v in cur_vel:
                if v[0] * x_dir < 0: v[0] = -v[0]
                if v[1] * y_dir < 0: v[1] = - v[1]
                prev_pos += v[0] * 0.01
            pos[i] = prev_pos
            prev_pos = pos[i] + 0
        ret[b] = pos
    return ret

def conver_to_straight_moving2(pos_, vel_):
    ret = np.zeros([2,19,2])
    for b in range(2):
        x_dir = 1
        y_dir = 1
        x_offset = 0
        y_offset = 0
        pos = pos_[b].reshape([19,2])
        vel = vel_[b].reshape([19,2])
        prev_pos = pos[0]
        prev_vel = vel[0]
        for i in range(1,19):
            cur_pos = pos[i]
            cur_vel = vel[i]
            # x 축으로 튕겼을 때
            if((cur_vel[0] >= 0 and prev_vel[0] < 0) or (cur_vel[0] < 0 and prev_vel[0] >= 0)):
                if prev_vel[0] * x_dir > 0:
                    x_offset += 1.8
                else:
                    x_offset -= 1.8
                x_dir *= -1
            if((cur_vel[1] >= 0 and prev_vel[1] < 0) or (cur_vel[1] < 0 and prev_vel[1] >= 0)):
                if prev_vel[1] * y_dir > 0:
                    y_offset += 1.8
                else:
                    y_offset -= 1.8
                y_dir *= -1
            pos[i][0] = x_offset + pos[i][0] * x_dir
            pos[i][1] = y_offset + pos[i][1] * y_dir

            prev_pos = cur_pos
            prev_vel = cur_vel
        ret[b] = pos
    return ret

def main(n=3, input_n = 10):
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # daita loader
    print('\n--- load dataset ---')
    if(origin_version): origin_root = "../datasets_npz/datasets_original_"+data_version
    else: origin_root = None

    dataset_test = dataset_pair_group(opts,"../datasets_npz/datasets_blur_"+data_version,"../datasets_npz/datasets_sharp_"+data_version, "../datasets_npz/datasets_sharp_start_"+data_version, "../datasets_npz/datasets_gt_vel_"+data_version, "../datasets_npz/datasets_gt_pos_"+data_version, "../datasets_npz/datasets_shutter_speed_"+data_version, test_split = test_split, origin_root=origin_root)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads)
    test_iter = iter(test_loader)

    if z_delete_mode:
        z_zero = torch.zeros(opts.batch_size, 32).cuda(opts.gpu)

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
    print('start the test at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)

    # test
    print('\n--- test ---')

    # for validation
    total_loss_vel = 0
    total_loss_gen_gt_vel = 0
    total_loss_content = 0
    total_loss_pred_vel = 0
    total_loss_pos = 0
    total_loss_ssim = 0
    total_loss_mse = 0
    loss_pos_per_sequence = np.zeros([opts.sequence_num - 2])
    loss_vel_per_sequence = np.zeros([opts.sequence_num - 2])
    loss_ssim_per_sequence = np.zeros([opts.sequence_num - 2])
    loss_converted_pos_per_sequence = np.zeros([n_obj, opts.sequence_num - 1])
    loss_mse_per_sequence = np.zeros([opts.sequence_num - 2])
    loss_mse_cov = np.array([])
    for i in range(n):
        # for validation
        loss_ssim = 0
        loss_vel = 0
        loss_gen_gt_vel = 0
        loss_content = 0
        loss_pred_vel = 0
        loss_pos = 0
        loss_mse = 0

        if not origin_version:
            try:
                images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s = next(test_iter)
            while(images_a_s.size(0) != opts.batch_size):
                try:
                    images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s = next(test_iter)
        else:
            try:
                images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s, images_origin_s = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s, images_origin_s = next(test_iter)
            while(images_a_s.size(0) != opts.batch_size):
                try:
                    images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s, images_origin_s = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    images_a_s, images_b_s, gt_s, gt_pos_s, images_a_start_s, shutter_speed_s, images_origin_s = next(test_iter)

        images_a_s = images_a_s.cuda(opts.gpu).detach()
        images_b_s = images_b_s.cuda(opts.gpu).detach()
        images_a_start_s = images_a_start_s.cuda(opts.gpu).detach()
        for o in range(n_obj):
            gt_s[o] = gt_s[o].cuda(opts.gpu).detach()
            gt_pos_s[o] = gt_pos_s[o].cuda(opts.gpu).detach()
        shutter_speed_s = shutter_speed_s.cuda(opts.gpu).detach()

        # for position prediction
        gt_pos_pred = np.zeros([n_obj, 2, opts.sequence_num-1, 2])
        gt_vel_pred = np.zeros([n_obj, 2, opts.sequence_num-1, 32])
        if(origin_version):
            images_origin_s = images_origin_s.cuda(opts.gpu).detach()
        # test model
        s = 0
        size = images_a_s.size()
        ret = torch.zeros(size)
        ret_gt = torch.zeros(size)
        ret_origin = torch.zeros(size)
        z = None
        ret[:,0] = images_b_s[:,0] # 첫번째 blur
        #ret[:,1] = images_b_s[:,1] # 두번째 blur
        ret_gt[:,0] = images_b_s[:,0] # 첫번째 blur
        ret_gt[:,1] = images_b_s[:,1] # 두번째 blur
        if origin_version:
            ret_origin[:,0] = images_origin_s[:,0] # 첫번째 blur
            ret_origin[:,1] = images_origin_s[:,1] # 두번째 blur
        # for velocity estimation
        vel_pred = 0
        mse_cov = np.zeros([2])

        while (s < (opts.sequence_num - 2)):

            if (s < input_n):
                images_sharp_prev = images_a_start_s[:,s,:,:,:]
                images_prev = images_b_s[:,s,:,:,:]
            else:
                images_sharp_prev = images_a_start_s[:,s,:,:,:]
                images_prev = model.input_B

            if ( (s+1) < input_n):
                images_a = images_a_start_s[:,s+1,:,:,:]
                images_b = images_b_s[:,s+1,:,:,:]
                images_a_end = images_a_s[:,s+1,:,:,:]
                given_vel = None
                given_pos = None


            else:
                images_a = images_a_start_s[:,s+1,:,:,:]
                images_b = model.post_B_pred_recons
                images_a_end = images_a_s[:,s+1,:,:,:]
                given_vel = model.next_vel_pred
                # Euler
                given_pos = model.next_pos_pred

            images_post = images_b_s[:,s+2,:,:,:]

            data_random = images_a_start_s[:,random.randint(0, opts.sequence_num - 1), :, :, :]
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

            if z_delete_mode:
                z = z_zero

            #model.test(images_sharp_prev, images_prev, images_post, images_a, images_b, gt, gt_pos, images_a_end, data_random, next_gt, next_gt_pos, z, given_vel = given_vel, given_pos = given_pos, shutter_speed=shutter_speed)
            model.test(images_prev, images_post, images_b, gt, gt_pos, next_gt, next_gt_pos, z, given_vel = given_vel, given_pos = given_pos, shutter_speed=shutter_speed)
            s += 1
            #z = None

            if ( (s+1) < input_n):
                ret[:,s+1] = images_b_s[:,s+1,:,:,:]
                ret_gt[:,s+1] = images_b_s[:,s+1,:,:,:]
                if(origin_version):
                    ret_origin[:,s+1] = images_origin_s[:,s+1,:,:,:]
                for o in range(n_obj):
                    gt_pos_pred[o,:,s-1] = model.pos_pred[o].cpu().detach()
                    gt_pos_pred[o,:,s] = model.next_pos_pred[o].cpu().detach()
                    gt_vel_pred[o,:,s-1] = model.vel_pred[o].cpu().detach()#[:,30:]
                    gt_vel_pred[o,:,s] = model.next_vel_pred[o].cpu().detach()#[:,30:]
            else:
                if (s+1) == input_n:
                    ret[:,s] = model.recons_B
                    #ret[:,s] = model.fake_B_encoded_with_gt[2]
                ret[:,s+1] = model.post_B_pred_recons
                ret_gt[:,s+1] = images_b_s[:,s+1,:,:,:]
                if(origin_version):
                    ret_origin[:,s+1] = images_origin_s[:,s+1,:,:,:]
                for o in range(n_obj):
                    gt_pos_pred[o,:,s-1] = model.pos_pred[o].cpu().detach()
                    gt_pos_pred[o,:,s] = model.next_pos_pred[o].cpu().detach()
                    gt_vel_pred[o,:,s-1] = model.vel_pred[o].cpu().detach()#[:,30:]
                    gt_vel_pred[o,:,s] = model.next_vel_pred[o].cpu().detach()#[:,30:]

            #for validation
            #loss_gen += model.loss_gen
            #loss_vel_recons += model.loss_vel_recons
            loss_gen_gt_vel += model.loss_gen_gt_vel
            #loss_inverse += model.loss_inverse
            #loss_vel_dir += model.loss_vel_dir
            #loss_content += model.loss_content
            loss_content = 0
            #loss_pred_vel = 0
            #loss_ssim += model.ssim_err
            for o in range(n_obj):
                loss_vel += model.loss_vel[o]
                loss_pred_vel += model.loss_pred_vel[o]
                loss_pos += model.loss_pos[o]

            #SSIM loss check version2 which is same method with in E3D
            #=============start from here======================
            pred_0 = model.post_B_pred_recons[0].cpu().detach().numpy().transpose([1,2,0])*0.5+0.5
            gt_0 = images_post[0].cpu().numpy().transpose([1,2,0])*0.5+0.5
            pred_1 = model.post_B_pred_recons[1].cpu().detach().numpy().transpose([1,2,0])*0.5+0.5
            gt_1 = images_post[1].cpu().numpy().transpose([1,2,0])*0.5+0.5

            pred_0_grey = pred_0[:,:,0:1] * 0.2126 + pred_0[:,:,1:2] * 0.7152 + pred_0[:,:,2:] * 0.0722
            gt_0_grey = gt_0[:,:,0:1] * 0.2126 + gt_0[:,:,1:2] * 0.7152 + gt_0[:,:,2:] * 0.0722
            pred_1_grey = pred_1[:,:,0:1] * 0.2126 + pred_1[:,:,1:2] * 0.7152 + pred_1[:,:,2:] * 0.0722
            gt_1_grey = gt_1[:,:,0:1] * 0.2126 + gt_1[:,:,1:2] * 0.7152 + gt_1[:,:,2:] * 0.0722

            loss_ssim += compare_ssim(pred_0_grey, gt_0_grey, multichannel=True, win_size =7)/2
            loss_ssim += compare_ssim(pred_1_grey, gt_1_grey, multichannel=True, win_size =7)/2

            loss_mse += np.sum((pred_0_grey - gt_0_grey)**2)/2
            loss_mse += np.sum((pred_1_grey - gt_1_grey)**2)/2
            mse_cov[0] += [np.sum((pred_0_grey - gt_0_grey)**2)]
            mse_cov[1] += [np.sum((pred_1_grey - gt_1_grey)**2)]
            #=============to here==============================

            loss_vel_per_sequence[s-1] += loss_vel/n
            loss_pos_per_sequence[s-1] += loss_pos/n
            loss_mse_per_sequence[s-1] += (np.sum((pred_0_grey - gt_0_grey)**2)/2 + np.sum((pred_1_grey - gt_1_grey)**2)/2)/n
            #loss_ssim_per_sequence[s-1] += model.ssim_err/n
            loss_ssim_per_sequence[s-1] += (compare_ssim(pred_0, gt_0, multichannel=True, win_size =7)/2 + compare_ssim(pred_1, gt_1, multichannel=True, win_size =7)/2)/n
            '''
            print("loss_vel_"+str(s)+": "+ str(model.loss_vel))
            print("loss_pos_"+str(s)+": "+ str(model.loss_pos))
            '''
        if data_version in convert_datasets:
            visited = [[], []] # batch size
            for o in range(n_obj):
                min = [1000, 1000]
                argmin = [0, 0] # batch size
                gt_pos_convert = gt_pos_s[o][:,1:,30:].cpu().numpy()
                gt_vel_convert = gt_s[o][:,1:,:].cpu().numpy()
                #print(gt_pos_convert.shape)
                for b in range(2): # batch size
                    for oo in range(n_obj):
                        if oo in visited[b]: continue
                        if np.mean( (gt_pos_convert[b][0] - gt_pos_pred[oo][b][0]) ** 2 ) < min[b]:
                            min[b] = np.mean( (gt_pos_convert[b][0] - gt_pos_pred[oo][b][0]) ** 2 )
                            argmin[b] = oo
                    visited[b].append(argmin[b])
                comp_pos_convert = np.zeros_like(gt_pos_pred[0])
                comp_vel_convert = np.zeros_like(gt_vel_pred[0])
                #print(comp_pos_convert.shape)
                for b in range(2):
                    comp_pos_convert[b] = gt_pos_pred[argmin[b], b]
                    comp_vel_convert[b] = gt_vel_pred[argmin[b], b]
                #print(gt_pos_convert)
                gt_pos_convert = conver_to_straight_moving(gt_pos_convert, gt_vel_convert)
                comp_pos_convert = conver_to_straight_moving(comp_pos_convert, comp_vel_convert)
                #print(gt_pos_convert)

                #print(gt_pos_convert[1])
                #print(comp_pos_convert[1])
                converted_pos_loss = np.mean((gt_pos_convert - comp_pos_convert)**2*100,axis=(0,-1))
                #print(converted_pos_loss)
                print("converted_pos_loss: " + str(np.mean(converted_pos_loss)))
                #loss_converted_pos_per_sequence += converted_pos_loss/n
                loss_converted_pos_per_sequence[o] += converted_pos_loss/n
        if throwing_mode:
            trajectories_gt = []
            trajectories_pred = []
            for o in range(n_obj):
                trajectories_gt.append(gt_pos_s[o][:,1:,30:].cpu().numpy())
                trajectories_pred.append(gt_pos_pred[o])
            img_dir = '%s/prediction' % (opts.visualize_root)
            img_filename = '/trajectory_%05d_%02d.jpg' % (ep0, i)
            show_trajectory(img_dir, img_filename, trajectories_gt, trajectories_pred)
        #conver_to_straight_moving
        #for validation
        #loss_gen /= (opts.sequence_num - 2)
        loss_vel /= ((opts.sequence_num - 2)*n_obj)
        #loss_vel_recons /= (opts.sequence_num - 2)
        loss_gen_gt_vel /= (opts.sequence_num - 2)
        #loss_inverse /= (opts.sequence_num - 2)
        #loss_vel_dir /= (opts.sequence_num - 2)
        #loss_content /= (opts.sequence_num - 2)
        loss_pred_vel /= ((opts.sequence_num - 2)*n_obj)
        loss_pos /= ((opts.sequence_num - 2)*n_obj)
        loss_ssim /= (opts.sequence_num - 2)
        #loss_mse /= (opts.sequence_num -2)
        print('=============================')
        print('vel_loss %04f, gen_loss_gt_vel %04f, loss_content %04f, vel_pred_loss %04f, pos_loss %04f, ssim_loss %04f, loss_mse %04f' % (loss_vel, loss_gen_gt_vel, loss_content, loss_pred_vel, loss_pos, loss_ssim, loss_mse))
        print('=============================')
        saver.write_pred_img(ep0, i, input_n, ret, ret_gt, ret_origin=ret_origin, origin_version = origin_version)

        #total_loss_gen += loss_gen
        total_loss_vel += loss_vel
        #total_loss_vel_recons += loss_vel_recons
        total_loss_gen_gt_vel += loss_gen_gt_vel
        #total_loss_inverse += loss_inverse
        #total_loss_vel_dir += loss_vel_dir
        total_loss_content += loss_content
        total_loss_pred_vel += loss_pred_vel
        total_loss_pos += loss_pos
        total_loss_ssim += loss_ssim.item()
        total_loss_mse += loss_mse
        loss_mse_cov = np.append(loss_mse_cov, mse_cov[0])
        loss_mse_cov = np.append(loss_mse_cov, mse_cov[1])


    #total_loss_gen /= n
    total_loss_vel /= n
    #total_loss_vel_recons /= n
    total_loss_gen_gt_vel /= n
    #total_loss_inverse /= n
    #total_loss_vel_dir /= n
    total_loss_content /= n
    total_loss_pred_vel /= n
    total_loss_pos /= n
    total_loss_ssim /= n
    total_loss_mse /= n

    print('==========================================================')
    print('===========================TOTAL==========================')
    print('vel_loss %04f, gen_loss_gt_vel %04f, loss_content %04f, vel_pred_loss %04f, pos_loss %04f, loss_ssim %04f, loss_mse %04f' % (total_loss_vel, total_loss_gen_gt_vel, total_loss_content, total_loss_pred_vel, total_loss_pos, total_loss_ssim, total_loss_mse))
    print("converted_pos_loss %04f" % (np.mean(loss_converted_pos_per_sequence)))
    print('==========================================================')
    print('==========================================================')
    print("loss_pos %04f"%total_loss_pos)
    for i in range(opts.sequence_num - 2):
        print(str(loss_pos_per_sequence[i]))
    print("loss_vel")
    for i in range(opts.sequence_num - 2):
        print(str(loss_vel_per_sequence[i]))
    print("loss_ssim %04f"%total_loss_ssim)
    for i in range(opts.sequence_num - 2):
        print(str(loss_ssim_per_sequence[i]))
    print("loss_mse %04f"%total_loss_mse)
    for i in range(opts.sequence_num - 2):
        print(str(loss_mse_per_sequence[i]))
    if data_version  in convert_datasets:
        print("loss_converted_pos %04f"% (np.mean(loss_converted_pos_per_sequence)))
        loss_converted_pos_per_sequence = np.mean(loss_converted_pos_per_sequence, axis = 0)
        for i in range(opts.sequence_num - 2):
            print(str(loss_converted_pos_per_sequence[i]))

    print("mse cov: " + str(np.cov(loss_mse_cov)))
    return

if __name__ == '__main__':
  main(n = 15, input_n = 2)
