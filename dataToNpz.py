import os
import numpy as np
import cv2
from PIL import Image
import imageio
import glob
import cv2
import random

data_folder = "UnrealData"

def pick_random(min_num, max_num):
    if (min_num == max_num):
        return min_num
    else:
        return random.random()*(max_num-min_num) + min_num
def g(array,gamma = 2.2, inverse = False):
    if(inverse):
        return (array**gamma)
    else:
        return (array**(1/gamma))

def make_blur(array, source_fps, dest_fps, w, h, shutter_speed_ratio = 1):
    if (source_fps % dest_fps != 0):
        print("make_blur function: fps 가 정확히 나누어 떨어지지 않습니다.")
        return 0
    M = source_fps // dest_fps
    array = g(array, inverse = True)
    cur_source_frame = 0
    length = array.shape[0] // M
    ret = np.zeros([length, h, w, 3], dtype = np.uint8)
    for i in range(length):
        frame = np.zeros([h,w,3])
        blured_frame_num = 0
        for m in range(M):
            if ((m/(M-1)) >= (0.5 - shutter_speed_ratio*0.5)) and ((m/(M-1)) <= (0.5 + shutter_speed_ratio*0.5)):
                frame += array[cur_source_frame]
                blured_frame_num += 1
            cur_source_frame += 1
        frame /= blured_frame_num
        frame = g(frame)
        ret[i] = frame
    return ret

def generate_blur_and_gt(data_num, h=64, w=64, seconds = 1/3, fps_1 = 960, fps_2 = 60, shutter_speed_random = False, min_shutter_speed_ratio = 0.5, max_shutter_speed_ratio = 1.0, rep = 1, data_version = "", whole_sharp_mode = False, obj_num=1):
    data_version = "_"+ data_version + ".npz"
    dest_blur = "./datasets_npz/datasets_blur" + data_version
    dest_sharp = "./datasets_npz/datasets_sharp" + data_version
    dest_sharp_start = "./datasets_npz/datasets_sharp_start" + data_version

    dest_gt = []
    dest_gt_pos = []
    for i in range(1,obj_num+1):
        if i ==1:
            dest_gt.append("./datasets_npz/datasets_gt_vel" + data_version)
            dest_gt_pos.append("./datasets_npz/datasets_gt_pos" + data_version)
        else:
            dest_gt.append("./datasets_npz/datasets_gt_vel" + str(i) + data_version)
            dest_gt_pos.append("./datasets_npz/datasets_gt_pos" + str(i) + data_version)

    dest_shutter_speed = "./datasets_npz/datasets_shutter_speed" + data_version
    if(whole_sharp_mode):
        dest_whole_sharp = "./datasets_npz/datasets_sharp_whole" + data_version
    # param
    data_length = int(seconds * fps_1)
    ratio = fps_1//fps_2
    # rep 완성하기.
    ret_blur_final = np.zeros([data_num*rep,data_length//ratio, h, w, 3], dtype=np.uint8)
    ret_sharp_final = np.zeros([data_num*rep,data_length//ratio, h, w, 3], dtype=np.uint8)
    ret_sharp_start_final = np.zeros([data_num*rep,data_length//ratio, h, w, 3], dtype=np.uint8)
    gt_final = []
    gt_pos_final = []
    for i in range(1,obj_num+1):
        gt_final.append(np.zeros([data_num*rep, data_length//ratio, ratio, 2], dtype=np.float))
        gt_pos_final.append(np.zeros([data_num*rep, data_length//ratio, ratio, 2], dtype=np.float))
    ret_shutter_speed_final = np.zeros([data_num*rep, 1], dtype=np.float)
    if(whole_sharp_mode):
        ret_sharp_whole_final = np.zeros([data_num*rep,data_length, h, w, 3], dtype=np.uint8)
    for cur_rep in range(rep):
        cur_data_num = 0

        sequence = np.zeros([data_length, h, w, 3], dtype=np.uint8)
        sequence_selected = np.zeros([data_length//ratio, h, w, 3], dtype=np.uint8)
        sequence_selected_start = np.zeros([data_length//ratio, h, w, 3], dtype=np.uint8)
        ret_blur = np.zeros([data_num,data_length//ratio, h, w, 3], dtype=np.uint8)
        # 끝부분 (16~31 일 때, 31)
        ret_sharp = np.zeros([data_num,data_length//ratio, h, w, 3], dtype=np.uint8)
        # 첫부분 (16~31 일 때, 16)
        ret_sharp_start = np.zeros([data_num,data_length//ratio, h, w, 3], dtype=np.uint8)
        gt = []
        gt_pos = []
        for i in range(1,obj_num+1):
            gt.append(np.zeros([data_num, data_length//ratio, ratio, 2], dtype=np.float))
            gt_pos.append(np.zeros([data_num, data_length//ratio, ratio, 2], dtype=np.float))
        ret_shutter_speed = np.zeros([data_num, 1], dtype=np.float)
        if(whole_sharp_mode):
            ret_sharp_whole = np.zeros([data_num,data_length, h, w, 3], dtype=np.uint8)
        try:
            if not(os.path.isdir("./datasets_npz")):
                os.makedirs(os.path.join("./datasets_npz"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
        for DataGroup_path in glob.glob("./"+data_folder+"/DataGroup*"):
            for Data_path in glob.glob(DataGroup_path+"/Data*"):
                # 이미지
                for sequence_n in range(1, data_length+1):
                    file_name = Data_path + "/sequence"+ str(sequence_n) + ".png"
                    image = imageio.imread(file_name)[:,:,:3] # 'RGBA' -> 'RGB'
                    image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                    sequence[sequence_n-1] = image
                    if(sequence_n % ratio == 0):
                        sequence_selected[sequence_n//ratio-1] = image
                    if(sequence_n % ratio == 1):
                        sequence_selected_start[sequence_n//ratio] = image
                sharp_datasets = sequence_selected
                if(not shutter_speed_random):
                    shutter_speed_ratio = 0.5
                else:
                    shutter_speed_ratio = pick_random(min_shutter_speed_ratio , max_shutter_speed_ratio)
                    #print(shutter_speed_ratio)
                blur_datasets = make_blur(sequence, 960, 60, w, h, shutter_speed_ratio)
                ret_sharp[cur_data_num] = sharp_datasets
                ret_sharp_start[cur_data_num] = sequence_selected_start
                ret_blur[cur_data_num] = blur_datasets
                ret_shutter_speed[cur_data_num] = shutter_speed_ratio
                if(whole_sharp_mode):
                    ret_sharp_whole[cur_data_num] = sequence
                # 위치, 속도
                for o in range(1,obj_num+1):
                    if o == 1:
                        file_name = Data_path + "/DagaGT.txt"
                    else:
                        file_name = Data_path + "/DagaGT" + str(o) + ".txt"
                    f = open(file_name,'r')

                    for i in range(data_length//ratio):
                        for k in range(ratio):
                            line = f.readline()
                            informs = line.split(" ") # pos_y pos_z vel_y vel_z
                            gt_pos[o-1][cur_data_num][i][k][0] = float(informs[0]) #pos_y
                            gt_pos[o-1][cur_data_num][i][k][1] = float(informs[1]) #pos_z
                            gt[o-1][cur_data_num][i][k][0] = float(informs[2]) #vel_y
                            gt[o-1][cur_data_num][i][k][1] = float(informs[3]) #vel_z
                            #print(gt[cur_data_num][i][k])
                            '''
                            # for y-positive gt
                            if(float(informs[2])<0):
                                gt[cur_data_num][i][k][0] = -gt[cur_data_num][i][k][0]
                                gt[cur_data_num][i][k][1] = -gt[cur_data_num][i][k][1]
                            '''
                cur_data_num += 1
                if (cur_data_num% 10 == 0):
                    print("진행률 "+str(cur_data_num+(cur_rep*data_num))+"/"+str(data_num*rep))
                if(cur_data_num >= data_num):
                    break
            if(cur_data_num >= data_num):
                break
        ret_blur_final[cur_rep*data_num:(cur_rep+1)*data_num] = ret_blur
        ret_sharp_final[cur_rep*data_num:(cur_rep+1)*data_num] = ret_sharp
        ret_sharp_start_final[cur_rep*data_num:(cur_rep+1)*data_num] = ret_sharp_start
        for o in range(1,obj_num+1):
            gt_final[o-1][cur_rep*data_num:(cur_rep+1)*data_num] = gt[o-1]
            gt_pos_final[o-1][cur_rep*data_num:(cur_rep+1)*data_num] = gt_pos[o-1]
        ret_shutter_speed_final[cur_rep*data_num:(cur_rep+1)*data_num] = ret_shutter_speed
        if(whole_sharp_mode):
            ret_sharp_whole_final[cur_rep*data_num:(cur_rep+1)*data_num] = ret_sharp_whole
    np.savez_compressed(dest_blur, ret_blur_final)
    np.savez_compressed(dest_sharp, ret_sharp_final)
    np.savez_compressed(dest_sharp_start, ret_sharp_start_final)
    for o in range(1,obj_num+1):
        np.savez_compressed(dest_gt[o-1], gt_final[o-1])
        np.savez_compressed(dest_gt_pos[o-1], gt_pos_final[o-1])
    np.savez_compressed(dest_shutter_speed, ret_shutter_speed_final)
    if(whole_sharp_mode):
        np.savez_compressed(dest_whole_sharp, ret_sharp_whole_final)

def generate_test(data_num, pic_num, h=64, w=64, seconds = 1, fps_1 = 960, fps_2 = 60):
    dest_blur = "./datasets_test.npz"
    cur_data_num = 0
    data_length = seconds * fps_1
    ratio = fps_1//fps_2
    ret_test = np.zeros([data_num,data_length//ratio, h, w, 3], dtype=np.uint8)
    try:
        if not(os.path.isdir("./testsets_npz")):
            os.makedirs(os.path.join("./testsets_npz"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    for DataGroup_path in glob.glob("./UnrealData/TEST"):
        for Data_path in glob.glob(DataGroup_path+"/Data*"):
            # 이미지
            while(cur_data_num<data_num):
                cur_data_num2 = 0
                while(cur_data_num2 < data_length//ratio):
                    for sequence_n in range(1, pic_num+1):
                        file_name = Data_path + "/sequence"+ str(sequence_n) + ".png"
                        image = imageio.imread(file_name)[:,:,:3] # 'RGBA' -> 'RGB'
                        image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                        ret_test[cur_data_num][cur_data_num2] = image;
                        cur_data_num2 += 1
                        if(cur_data_num2 == data_num):
                            break
                cur_data_num += 1
                if (cur_data_num% 10 == 0):
                    print("진행률 "+str(cur_data_num)+"/"+str(data_num))
    np.savez_compressed(dest_blur, ret_test)

def __main__():
    print("data generate")
    #generate_blur_and_gt(5000, shutter_speed_random=True, rep =2, data_version = "Euler")
    #generate_blur_and_gt(5000, h = 120, w = 160, seconds = 160/960, shutter_speed_random=True, rep =1, data_version = "real_restricted_more_realistic2")
    #generate_blur_and_gt(5000, h = 64, w = 64, seconds = 320/960, shutter_speed_random=False, rep =1, data_version = "final_drift_small", whole_sharp_mode = True)
    #generate_blur_and_gt(1000, h = 64, w = 64, seconds = 320/960, shutter_speed_random=True, rep =1, data_version = "final_drift_test_1000_after_400frames", whole_sharp_mode = False)
    #generate_blur_and_gt(5000, h = 64, w = 64, seconds = 320/960, shutter_speed_random=True, rep =1, data_version = "chair", whole_sharp_mode = False)
    #generate_blur_and_gt(3000, h = 64, w = 64, seconds = 320/960, shutter_speed_random=True, rep =1, data_version = "2balls_NoCollision_SameGravity", whole_sharp_mode = False, obj_num=2)
    generate_blur_and_gt(5000, h = 120, w = 160, seconds = 160/960, shutter_speed_random=True, rep =1, data_version = "  2balls_towards_wall_slow", whole_sharp_mode = False, obj_num=2)
    #generate_test(500,3)
if(__name__=="__main__"):
   __main__()
