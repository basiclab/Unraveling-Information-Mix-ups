import os 
import torch
import torch.nn.functional as F

from model.TEBOpt_pipeline import TEBPipeline
from model.attentions_utils import register_attention_control, AttentionStore
from torchvision.utils import save_image
from torchvision.io import read_image
from typing import List

import argparse
from PIL import Image
import numpy as np
import cv2

import utils.utils as utils
import time

def decode_tensors(pipe, step, callback_kwargs):
    # Display image after each generation step | reference: https://huggingface.co/docs/diffusers/using-diffusers/callback
    dir_name = os.path.join(args.save_dir, "step_imgs")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    latents = callback_kwargs["latents"]
    
    # latent to image
    image = pipe.latent2image(latents, return_type="np")
    image = Image.fromarray(image)
    attn_np = np.array(image.resize((512, 512)))
    img_np = cv2.cvtColor(attn_np, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(os.path.join(dir_name, "{}.png".format(step)), img_np)

    return callback_kwargs

def SD_init():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load SD 1.4
    model_path = "CompVis/stable-diffusion-v1-4"
    pipe = TEBPipeline.from_pretrained(model_path).to(device)
    print("load SD 1.4 pretrained model")

    return pipe

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    
    attention_maps = attention_store.get_average_attention()
    # attention_maps = attention_store.get_cur_attention()
    out = []
    num_pixels = res ** 2

    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)

    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]

    return out.cpu()

def show_cross_attention(seed, image_size, save_path, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, prefix=""):
    image_c, image_h, image_w = image_size
    tokens = pipe.tokenizer.encode(prompts[select])
    decoder =  pipe.tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    
    # Concatenate the images and write the corresponding word
    # Draw the words on the canvas
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_offset, y_offset = 0, 200
    font_scale = 2
    font_thickness = 2
    tar_h, tar_w = image_h, image_w
    text_shift_x, text_shift_y = 100, 75
    
    # the len(tokens) = len(prompts[0]) + 2 (startoftext and endoftext)
    for i in range(len(tokens)):
        image_tensor = attention_maps[:, :, i]
        h,w = image_tensor.shape   # h=16, w=16
        # convert tensor to numpy
        if h != tar_h or w != tar_w:
            image_tensor = utils.resize_net_attn_map(image_tensor, (tar_w, tar_h))
        attn_np = utils.attn_map_tr2np(image_tensor.numpy())
        h,w = attn_np.shape 
        token_id = tokens[i]
        token = decoder(int(tokens[i]))
        if i == 0:
            canvas = np.zeros((h + y_offset, w * (len(tokens)+1), 3), dtype=np.uint8)
            # paste the generated result as the first image
            result = cv2.imread(os.path.join(save_path, "{}{}_{}.png".format(prefix, prompts[0], seed)))
            result = cv2.resize(result, (tar_w, tar_h))
            canvas[y_offset:y_offset+h, x_offset:x_offset + w] = result
            cv2.putText(canvas, "Generated result", (x_offset + text_shift_x, text_shift_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            x_offset += w
        # convert PIL.Image to cv2
        img_np = cv2.cvtColor(attn_np, cv2.COLOR_RGB2BGR)
        # Write the token on result
        cv2.putText(canvas, token, (x_offset + text_shift_x, text_shift_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(canvas, str(token_id), (x_offset + text_shift_x, text_shift_y*2), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        # Paste cross-attention map on result
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img_np
        x_offset += w
    
    cv2.imwrite(os.path.join(save_path, "cross_attn", "{}{}_{}.png".format(prefix, prompts[select], seed)), canvas)

def show_self_attention(seed, image_size, save_path, prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0, prefix=""):
    image_c, image_h, image_w = image_size
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    
    if attention_maps.dtype == np.float16:
        attention_maps = attention_maps.astype(np.float32)
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    tar_h, tar_w = image_h, image_w

    for top_shift in [0]:#, 10, 20, 30, 40, 50]:
        images_vh, image_u = [], []
        for i in range(max_com):
            if i == 0:
                # paste the generated result as the first image
                result = Image.open(os.path.join(save_path, "{}{}_{}.png".format(prefix, prompts[0], seed))).resize((tar_h, tar_w))
                image = np.array(result)
                image_u.append(image)
            image = u[:, i+top_shift].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((tar_h, tar_w))
            image = np.array(image)
            image_u.append(image)

        avg_np_u = np.concatenate(image_u, axis=1)
        cv2.imwrite(os.path.join(save_path, "self_attn", "{}{}_{}.png".format(prefix, prompts[select], seed)), cv2.cvtColor(avg_np_u, cv2.COLOR_RGB2BGR))      
        
def gen_img(save_path, prompt_1, seed, concat_pure_text_emb, text_emb_optimize, indices_to_balance, masking_token_emb, masking_token_index, calcaluate_distance, prefix):

    utils.seed_everything(seed)

    prompts = [prompt_1]

    controller = AttentionStore()
    register_attention_control(pipe, controller)

    results = pipe(prompts,
                   seed=seed,
                    attention_store=controller,
                    guidance_scale=7.5,
                    save_path=save_path,
                    concat_pure_text_emb=concat_pure_text_emb,
                    text_emb_optimize=text_emb_optimize, 
                    indices_to_balance=indices_to_balance,
                    masking_token_emb=masking_token_emb,
                    masking_token_index=masking_token_index,
                    calcaluate_distance=calcaluate_distance,
                    # callback_on_step_end=decode_tensors,
                    # callback_on_step_end_tensor_inputs=["latents"],
                    )[0]
    if not calcaluate_distance:
        save_image(results, os.path.join(save_path, '{}{}_{}.png'.format(prefix, prompts[0], seed)))   
        image_size = results.squeeze(0).size() 
        # print("image_size after squeeze = ", image_size)    
        res = 16     

        # show_cross_attention(seed, image_size, save_path, prompts, controller, res, ["up", "down"], prefix=prefix)
        # show_self_attention(seed, image_size, save_path, prompts, controller, res, ["up", "down"],prefix=prefix)        

        # utils.gen_gif(src_dir= os.path.join(save_path, "step_imgs")) 


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(description="Code for Text Embedding Balance Optimization")
    parser.add_argument("--prompt", type=str, default="A cat and a dog", help="input prompt")
    parser.add_argument("--data_dir", type=str, default="./data/sample.txt", help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default="result/test", help="Path to the result directory")
    parser.add_argument("--seed", type=int, default=2, help="seed for generator")
    parser.add_argument("--concat_pure_text_emb", action='store_true', help="[Hypothesis] concat objects' pure text embeding")
    parser.add_argument("--text_emb_optimize", action='store_true', help="[Method] update object's text embedding")
    parser.add_argument("--indices_to_balance", type=str, default="2,5", help="[Method] indice selected for balance")
    parser.add_argument("--masking_token_emb", action='store_true', help="[analysis] masking text embedding")
    parser.add_argument("--masking_token_index", type=str, default="1,4", help="[analysis] indice selected for masking, the first number represents the starting index and second number represents the ending index")
    parser.add_argument("--calcaluate_distance", action='store_true', help="[eval] calculate the text embedding similarity and cross-attention map distance")
    
    args = parser.parse_args()
    
    args.save_dir = args.save_dir + "_" + os.path.basename(args.data_dir).split(".")[0]
    args.masking_token_index = [int(num) for num in args.masking_token_index.split(',')]
    args.indices_to_balance = [int(num) for num in args.indices_to_balance.split(',')]

    if args.calcaluate_distance:
        if args.text_emb_optimize:
            args.save_dir = args.save_dir + "_TEBOpt_cal_dist"
        else:
            args.save_dir = args.save_dir + "_cal_dist"
    if args.text_emb_optimize:
        args.save_dir = args.save_dir + "_TEBOpt"
    elif args.concat_pure_text_emb:
        args.save_dir = args.save_dir + "_Hypo1"
    elif args.masking_token_emb:
        args.save_dir = args.save_dir + "_mask{}:{}".format(args.masking_token_index[0], args.masking_token_index[1])

    print("args.save_dir = ", args.save_dir)

    if not os.path.exists(os.path.join(args.save_dir, "cross_attn")):
        os.makedirs(os.path.join(args.save_dir, "cross_attn"))
    if not os.path.exists(os.path.join(args.save_dir, "self_attn")):
        os.makedirs(os.path.join(args.save_dir, "self_attn"))

    start_time = time.time()

    pipe = SD_init()

    with open(args.data_dir, 'r') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            args.seed, args.prompt = line.split(";")
            print("seed = {}, prompt = {}, concat_pure_text_emb = {}, args.text_emb_optimize = {}, args.indices_to_balance = {}, args.masking_token_emb = {}, args.masking_token_index = {}, args.calcaluate_distance = {}".
                  format(args.seed, args.prompt, args.concat_pure_text_emb, args.text_emb_optimize, args.indices_to_balance, args.masking_token_emb, args.masking_token_index, args.calcaluate_distance))
            
            gen_img(args.save_dir, args.prompt, seed=args.seed, 
                    concat_pure_text_emb=args.concat_pure_text_emb, text_emb_optimize=args.text_emb_optimize,
                    indices_to_balance=args.indices_to_balance, masking_token_emb=args.masking_token_emb, masking_token_index=args.masking_token_index, calcaluate_distance=args.calcaluate_distance, prefix="")#
            torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time/(idx+1)} seconds")