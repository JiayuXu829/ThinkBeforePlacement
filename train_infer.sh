#!/bin/bash
#train
cd ~/projects/LLaVA_ECCV2025
sh scripts/image_comp_finetune_lora_eccv.sh

#inference
cd ~/projects/LLaVA_ECCV2025/llava/eval
CUDA_VISIBLE_DEVICES=3 python run_llava_generate_all_gussian.py --model_path /home/lm3/projects/ep0.5/llava-vicuna-7b-v1.3-finetune_lora_1 --output_path /home/lm3/projects/LLaVA_ECCV2025/rebuttal_res/ --epoch 3 --pred_num 10
CUDA_VISIBLE_DEVICES=3 python run_llava_generate_all_gussian.py --model_path /home/lm3/projects/ep0.5/llava-vicuna-7b-v1.3-finetune_lora_2 --output_path /home/lm3/projects/LLaVA_ECCV2025/rebuttal_res/ --epoch 3 --pred_num 10
CUDA_VISIBLE_DEVICES=3 python run_llava_generate_all_gussian.py --model_path /home/lm3/projects/ep0.5/llava-vicuna-7b-v1.3-finetune_lora_3 --output_path /home/lm3/projects/LLaVA_ECCV2025/rebuttal_res/ --epoch 3 --pred_num 10
