
#### 8 to4 + 4to 2 + 2 to 1
#python main.py --model EDSR --scale 8 --save EDSR_Fulltrain_x8_1_0211 --reset --pre_train_1 ../experiment/EDSR_baseline_x2_1/model/model_183.pt  --pre_train_2 ../experiment/EDSR_baseline_x4_2/model/model_268.pt --pre_train_3 ../experiment/EDSR_baseline_x8_4/model/model_297.pt














#python main.py --model EDSR --scale 4 --save EDSR_bicubic_Final_x4_2_0318_retrain22 --reset --chop_forward   --pre_train_1 ../experiment/EDSR_bicubic_Final_x4_2_0315_update/model/model_74.pt

#--pre_train_1 ../experiment/EDSR_bicubic_Final_x4_2_0315/model/model_53.pt # --pre_train_2 ../experiment/EDSR_baseline_x4_2/model/model_268.pt --pre_train_3 ../experiment/EDSR_bicubic_baseline_x8_4_0315_2/model/model_26.pt


# Test your own images
python main.py --scale 4 --data_test MyImage --test_only --save_results --pre_train_1 ../experiment/EDSR_bicubic_Final_x4_2_0318_retrain/model/model_63.pt   --self_ensemble







#Advanced - JPEG artifact removal
#python main.py --template MDSR_jpeg --model MDSR --scale 2+3+4 --save MDSR_jpeg --quality 75+ --reset
