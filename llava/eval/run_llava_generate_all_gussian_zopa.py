import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

import numpy as np
import csv
import os
import json
from tqdm import tqdm
from torchvision import transforms

import cv2
from skimage import io, img_as_float
from skimage.util import random_noise

# img_folder = "/home/lm3/data/new_OPA"
img_folder = "/home/lm3/projects/ZOPA_dataset"

# eval_data_path = "/home/lm3/data/new_OPA/trans_labels_gt_eval_set.json"
eval_data_path = "/home/lm3/projects/ZOPA_dataset/trans_labels_gt_eval_set.json"

def img_crop(x, x_mode, bbox):
    assert (x_mode in ['gray', 'color'])
    h_low, h_high, w_low, w_high = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
    y_arr = np.array(x, dtype=np.uint8)
    if x_mode == 'gray':
        y_arr = y_arr[h_low:h_high, w_low:w_high]
    else:
        y_arr = y_arr[h_low:h_high, w_low:w_high, :]
    y = Image.fromarray(y_arr)
    return y

def gen_composite_image(bg_img, fg_img, fg_msk, trans, fg_bbox=None):
    def modify(x, y, w, h):
        if x < 0:
            x = 0
        if x >= bg_img.size[0]:
            x = bg_img.size[0] - 1
        if y < 0:
            y = 0
        if y >= bg_img.size[1]:
            y = bg_img.size[1] - 1
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1
        return x, y, w, h
    if fg_bbox != None:
        fg_img = img_crop(fg_img, 'color', fg_bbox)
        fg_msk = img_crop(fg_msk, 'gray', fg_bbox)
    bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
    relative_scale, relative_x, relative_y = trans[0], trans[1], trans[2]
    if bg_w / bg_h > fg_w / fg_h:
        fg_w_new, fg_h_new = bg_h * relative_scale * fg_w / fg_h, bg_h * relative_scale
    else:
        fg_w_new, fg_h_new = bg_w * relative_scale, bg_w * relative_scale * fg_h / fg_w
    start_x, start_y, width, height = round((bg_w - fg_w_new) * relative_x), round((bg_h - fg_h_new) * relative_y), round(fg_w_new), round(fg_h_new)
    start_x, start_y, width, height = modify(start_x, start_y, width, height)
    resize_func = transforms.Resize((height, width), interpolation=Image.BILINEAR)
    fg_img_new, fg_msk_new = resize_func(fg_img), resize_func(fg_msk)
    comp_img_arr, bg_img_arr, fg_img_arr, fg_msk_arr = np.array(bg_img), np.array(bg_img), np.array(fg_img_new), np.array(fg_msk_new)
    fg_msk_arr_norm = fg_msk_arr[:,:,np.newaxis].repeat(3, axis=2) / 255.0
    comp_img_arr[start_y:start_y+height, start_x:start_x+width, :] = fg_msk_arr_norm * fg_img_arr + (1.0 - fg_msk_arr_norm) * bg_img_arr[start_y:start_y+height, start_x:start_x+width, :]
    comp_img = Image.fromarray(comp_img_arr.astype(np.uint8)).convert('RGB')
    comp_msk_arr = np.zeros(comp_img_arr.shape[:2])
    comp_msk_arr[start_y:start_y+height, start_x:start_x+width] = fg_msk_arr
    comp_msk = Image.fromarray(comp_msk_arr.astype(np.uint8)).convert('L')
    return comp_img, comp_msk, [start_x, start_y, width, height]

def get_cate_map(root_directory:str)->dict:
    category_image_dict = dict()
    # 遍历根目录下的子目录，每个子目录代表一个类别
    for category in os.listdir(root_directory):
        category_directory = os.path.join(root_directory, category)
        
        # 检查是否是目录
        if os.path.isdir(category_directory):
            
            # 遍历类别目录下的图像文件
            for image_filename in os.listdir(category_directory):
                # 将图像文件名和类别添加到字典中
                category_image_dict[image_filename] = category
    return category_image_dict

def csv_title():
    return 'annID,scID,bbox,catnm,label,img_path,msk_path'

def csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name):
    return '{},{},"{}",{},-1,images/{}.jpg,masks/{}.png'.format(annid, scid, gen_comp_bbox, catnm, gen_file_name, gen_file_name)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    output_dict_list = list()
    gen_res_list = list()
    pred_num = args.pred_num

    for i in range(pred_num):
        output_dict_list.append(list())
        gen_res_list.append(list())
    # cate_map = get_cate_map("/home/lm3/data/new_OPA/foreground")
    cate_map = get_cate_map("/home/lm3/projects/ZOPA_dataset/foreground")
    #set datapath 
    model_name = get_model_name_from_path(args.model_path)
    save_dir = os.path.join(args.output_path, model_name) 
    save_dir_list = [os.path.join(save_dir,f"pred{i}") for i in range(pred_num)]
    eval_dir_list = [os.path.join(i,args.eval_type,str(args.epoch)) for i in save_dir_list]
    for eval_dir in eval_dir_list:
        assert (not os.path.exists(eval_dir))
    
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    with open(eval_data_path, "r") as f:
        eval_data_infos = json.load(f)
        for index,data_info in tqdm(enumerate(eval_data_infos)):
            fg_img_filename = data_info['fg_img_filename']
            bg_img_filename = data_info['bg_img_filename']
            mask_fg_img_filename = data_info['mask_fg_img_filename']

            fg_img = load_image(os.path.join(img_folder, "no_class_foreground", fg_img_filename))
            mask_fg_img = load_image(os.path.join(img_folder, "no_class_foreground", mask_fg_img_filename))
            bg_img = load_image(os.path.join(img_folder, "no_class_background", bg_img_filename))
            masked_fg_img = Image.fromarray(np.array(fg_img) * np.array(mask_fg_img))

            noisy_masked_fg_img = random_noise(img_as_float(np.array(masked_fg_img)), mode='gaussian', var=1.5)
            noisy_bg_img = random_noise(img_as_float(np.array(bg_img)), mode='gaussian', var=1.5)
            noisy_masked_fg_img_uint8 = (noisy_masked_fg_img * 255).astype(np.uint8)
            noisy_bg_img_uint8 = (noisy_bg_img * 255).astype(np.uint8)

            masked_fg_img_tensor = image_processor.preprocess(Image.fromarray(noisy_masked_fg_img_uint8), return_tensors='pt')['pixel_values'].half().cuda()    
            bg_img_tensor = image_processor.preprocess(Image.fromarray(noisy_bg_img_uint8), return_tensors='pt')['pixel_values'].half().cuda()

            images = torch.stack([masked_fg_img_tensor, bg_img_tensor])
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=30,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs_caption = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs_caption = outputs_caption.strip()
            if outputs_caption.endswith(stop_str):
                outputs_caption = outputs_caption[:-len(stop_str)]
            outputs_caption = outputs_caption.strip()
            
            new_prompt = prompt + outputs_caption
            input_ids = tokenizer_image_token(new_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                pred_trans = model.generate_trans_label_all(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )
            for pred_idx in range(pred_trans.shape[1]):
                pred_trans_single = pred_trans[:,pred_idx]
                output_dict = output_dict_list[pred_idx]
                eval_dir = eval_dir_list[pred_idx]
                save_dir = save_dir_list[pred_idx]
                gen_res = gen_res_list[pred_idx]
                output_dict.append(
                    dict(
                        fg_img_filename = fg_img_filename,
                        bg_img_filename = bg_img_filename,
                        caption = outputs_caption,
                        pred_trans = pred_trans_single.tolist(),
                    )
                )
                bg_img_arr = np.array(bg_img)
                fg_img_arr = np.array(fg_img)
                fg_msk_arr = np.array(mask_fg_img)

                img_sav_dir = os.path.join(eval_dir, 'images')
                msk_sav_dir = os.path.join(eval_dir, 'masks')
                # csv_sav_file = os.path.join(eval_dir, '{}.csv'.format(args.eval_type))
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)
                if not os.path.exists(img_sav_dir):
                    os.mkdir(img_sav_dir)
                if not os.path.exists(msk_sav_dir):
                    os.mkdir(msk_sav_dir)
                for repeat_id in range(args.repeat):
                    gen_comp_img, gen_comp_msk, gen_comp_bbox = gen_composite_image(
                        bg_img=Image.fromarray(bg_img_arr.astype(np.uint8)).convert('RGB'), 
                        fg_img=Image.fromarray(fg_img_arr.astype(np.uint8)).convert('RGB'), 
                        fg_msk=Image.fromarray(fg_msk_arr.astype(np.uint8)).convert('L'), 
                        trans=(pred_trans_single.cpu().numpy().astype(np.float32)[0]).tolist(),
                        fg_bbox=None
                    )
                    annid = fg_img_filename.strip('.jpg')
                    scid = bg_img_filename.strip('.jpg')
                    if args.repeat == 1:
                        gen_file_name = "{}_{}_{}_{}_{}_{}_{}".format(index, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
                    else:
                        gen_file_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(index, repeat_id, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])

                    gen_comp_img.save(os.path.join(img_sav_dir, '{}.jpg'.format(gen_file_name)))
                    gen_comp_msk.save(os.path.join(msk_sav_dir, '{}.png'.format(gen_file_name)))
                    gen_res.append(csv_str(annid, scid, gen_comp_bbox, cate_map[fg_img_filename], gen_file_name))
        
        for pred_idx, gen_res in enumerate(gen_res_list):
            eval_dir = eval_dir_list[pred_idx]
            save_dir = save_dir_list[pred_idx]
            output_dict = output_dict_list[pred_idx]
            csv_sav_file = os.path.join(eval_dir, '{}.csv'.format(args.eval_type))
            output_filepath = os.path.join(save_dir, 'pred_trans_test.json')
            
            with open(csv_sav_file, "w") as f:
                f.write(csv_title() + '\n')
                for line in gen_res:
                    f.write(line + '\n')
            with open(output_filepath, "w") as f:
                json.dump(output_dict,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m", required=True)
    parser.add_argument("--model_base", default="/home/lm3/checkpoints/vicuna-7b-v1.3", type=str)
    parser.add_argument("--query", default="Where would it be reasonable to place the object from first image in the second image?", type=str)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="/home/lm3/projects/LLaVA_ECCV2025/rebuttal_res/", required=False)
    parser.add_argument("--eval_type", type=str, default="eval")
    parser.add_argument("--repeat", type=str, default=1)
    parser.add_argument("--epoch", type=int,  required=True)
    parser.add_argument("--pred_num", type=int, default=None, required=True)
    
    args = parser.parse_args()

    eval_model(args)
