# EDSR baseline model (x2)
#python main.py --model EDSR --scale 2 --save EDSR_baseline_x2 --reset

# EDSR baseline model (x3) - requires pre-trained EDSR baseline x2 model
#python main.py --model EDSR --scale 3 --save EDSR_baseline_x3 --reset --pre_train ../experiment/model/EDSR_baseline_x4.pt

# EDS R baseline model (x4) - requires pre-trained EDSR baseline x2 model
#### 8 to 1
#python main.py --model EDSR --scale 8 --save EDSR_Fulltrain_x8_1_0210 --reset --pre_train_1 ../experiment/EDSR_baseline_x2_1/model/model_183.pt  --pre_train_2 ../experiment/EDSR_baseline_x4_2/model/model_268.pt --pre_train_3 ../experiment/EDSR_baseline_x8_4/model/model_297.pt


#### 8 to4 + 4to 2 + 2 to 1
#python main.py --model EDSR --scale 8 --save EDSR_Fulltrain_x8_1_0211 --reset --pre_train_1 ../experiment/EDSR_baseline_x2_1/model/model_183.pt  --pre_train_2 ../experiment/EDSR_baseline_x4_2/model/model_268.pt --pre_train_3 ../experiment/EDSR_baseline_x8_4/model/model_297.pt


#python main.py --model EDSR --scale 1 --save DNEDSR__x4_4_0315_track2_ResDnCNN_L1_Final --reset --chop_forward  # --pre_train_1 ../experiment/DEDSR__x4_4_0315_track2_Feel/model/model_30.pt #  --pre_train_1 ../experiment/EDSR_baseline_x2_1/model/model_183.pt  --pre_train_2 ../experiment/EDSR_baseline_x4_2/model/model_268.pt --pre_train_3 ../experiment/EDSR_baseline_x8_4/model/model_297.pt
#python main.py --model EDSR --scale 1 --save DEDSR__x4_2_0312_track2_Batch_3 --reset --pre_train_1 ../experiment/DEDSR__x4_4_0312_track2_No_batch/model/model_1.pt  --pre_train_2 ../experiment/EDSR_baseline_x4_2/model/model_268.pt  --chop_forward


#python main.py --save EDSR_baseline_x8_4 --reset --fullInputScale 8 --fullTargetScale 4 --patch_mult 0.5 --n_GPU_number 2 #0123

#python main.py --save EDSR_baseline_x4_2 --reset --fullInputScale 4 --fullTargetScale 2 --patch_mult 0.5 --n_GPU_number 1 #0123

#python main.py --save EDSR_baseline_x2_1 --reset --fullInputScale 2 --fullTargetScale 1 --patch_mult 1 --n_GPU_number 3 #0123



# Test your own images
python main.py --scale 1 --data_test MyImage --test_only --save_results --pre_train_1 ../experiment/DNEDSR__x4_4_0315_track2_ResDnCNN_L1_Final/model/model_236.pt --self_ensemble

#Advanced - JPEG artifact removal
#python main.py --template MDSR_jpeg --model MDSR --scale 2+3+4 --save MDSR_jpeg --quality 75+ --reset
