pred_mode =True
gen_mode = True
pos_gen_mode = True
B0minusB1Mode = False
origin_version = False
gen_manual_mode = False


# test 단계에서는 아래 2 변수 상관 없다.
vel_secret_mode = False # prediction net tuning mode
encoder_tuning_mode = False
sharp_whole_mode = False #vel_secret_mode에서만 활성화 가능

#z variable 유무
z_delete_mode = True

#sharp loss 유무 (train 이고, sharp_whole_mode 가 꺼져있을 때만 의미 있음)
sharp_loss_mode = False

#sharp gen 유무
sharp_gen_mode =False

#last_vel_loss 유무
last_vel_loss_mode = False

#test시 이미지 하나마다 저장
one_by_one_save_mode = True

# velocity encoder가 dir, vec으로 나눠짐
dir_mode = True

# pred net 안에서, pos, vel 을 단순히 concat하는 것이 아니라 일련의 pos로 바꾼 후 넣음.
pred2_mode = True
pred_relation_mode = False

network2_mode = True

# Throwing mode에서는 initial 2 frames에서만 encoder, gen을 학습하고, 이후로는 prediction network만 학습한다.
throwing_mode = False
n_obj = 1
