import os
from PIL import Image, ImageDraw, ImageFont
import torch
import os
import json
import glob
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import time
from tqdm import tqdm

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

"""
{
    'identity': (
            "2 object exist": 0, 
            "mixtures": 0, 
            "obj_1 + mixture": 0,
            "mixture + obj_2": 0, 
            "only obj_1 exists (>1)": 0,
            "only obj_1 exists (=1)": 0,
            "only obj_2 exists (>1)": 0,
            "only obj_2 exists (=1)": 0,
            "no target object": 0
        ),
    'counts': "n {first_half}, n {second_half}, n {mixture}"
}
"""
def check_mixture(s_json, first_half, second_half):
    result = dict()
    if(len(s_json['objects'].keys()) == 0):
        result['identity'] = 'no target object'
        result['counts'] = '0'
        return result
    
    if(len(s_json['objects'].keys()) == 1):
        if first_half in s_json['objects'].keys():
            if len(s_json['objects'][first_half]) > 1:
                result['identity'] = 'only obj_1 exists (>1)'
            else:
                result['identity'] = 'only obj_1 exists (=1)'
            result['counts'] = f"{len(s_json['objects'][first_half])} {first_half}"
        else:
            if len(s_json['objects'][second_half]) > 1:
                result['identity'] = 'only obj_2 exists (>1)'
            else:
                result['identity'] = 'only obj_2 exists (=1)'
            result['counts'] = f"{len(s_json['objects'][second_half])} {second_half}"
        return result

    first_object_set = set(obj[0] for obj in s_json['objects'][first_half])
    sec_object_set = set(obj[0] for obj in s_json['objects'][second_half])
    mixture_num = 0
    for first in s_json['objects'][first_half]:
        idx1, box1, conf1 = first
        if conf1 < 0.4 and idx1 in first_object_set:
            first_object_set.remove(idx1)
        for sec in s_json['objects'][second_half]:
            idx2, box2, conf2 = sec
            if conf2 < 0.4 and idx2 in sec_object_set:
                sec_object_set.remove(idx2)
            box1_x1, box1_y1, box1_x2, box1_y2 = box1
            box2_x1, box2_y1, box2_x2, box2_y2 = box2

            x_left = max(box1_x1, box2_x1)
            y_top = max(box1_y1, box2_y1)
            x_right = min(box1_x2, box2_x2)
            y_bottom = min(box1_y2, box2_y2)
            if x_right < x_left or y_bottom < y_top:
                continue
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            if iou > 0.9:
                mixture_num += 1
                if idx1 in first_object_set:
                    first_object_set.remove(idx1)
                if idx2 in sec_object_set:
                    sec_object_set.remove(idx2)
    if mixture_num > 0 and len(first_object_set) == 0 and len(sec_object_set) == 0:
        result['identity'] = "mixtures"
    elif len(first_object_set) > 0 and len(sec_object_set) > 0:
        result['identity'] = "2 object exist"
        if mixture_num > 0:
            result['identity'].replace("exist", " + mixture")
    elif mixture_num > 0:
        result['identity'] = ""
        if len(first_object_set) > 0:
            result['identity'] = "obj_1 + mixture"
        elif len(sec_object_set) > 0:
            result['identity'] = "mixture + obj_2"
    elif len(first_object_set) > 0:
        if len(first_object_set) > 1:
            result['identity'] = 'only obj_1 exists (>1)'
        else:
            result['identity'] = 'only obj_1 exists (=1)'
    elif len(sec_object_set) > 0:
        if len(sec_object_set) > 1:
            result['identity'] = 'only obj_2 exists (>1)'
        else:
            result['identity'] = 'only obj_2 exists (=1)'
    else:
        result['identity'] = "no target object"

    
    first_half_str = f"{len(first_object_set)} {first_half}"
    second_half_str = f"{len(sec_object_set)} {second_half}"
    result['counts'] = f"{first_half_str}, {second_half_str}, {mixture_num} mixture"
    return result

def owlv2_eval(exp_name='test_sample', image_path="", save_path=""):
    colors = Colors()  # create instance for 'from utils.plots import colors'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

    img_path = "owl_v2_imgs"
    label_path = "owl_v2_labels"

    os.makedirs(os.path.join(save_path, img_path), exist_ok=True)
    os.makedirs(os.path.join(save_path, label_path), exist_ok=True)

    # for sd_type in sd_types:
    os.makedirs(os.path.join(save_path, img_path, exp_name), exist_ok=True)
    os.makedirs(os.path.join(save_path, label_path, exp_name), exist_ok=True)

    start_time = time.time()
    eval_num = len(glob.glob(os.path.join(image_path, "*.png")))
    result_list = {}
    summary = {
        'identity_counts': {
            "2 object exist": 0, 
            "mixtures": 0, 
            "obj_1 + mixture": 0,
            "mixture + obj_2": 0, 
            "only obj_1 exists (>1)": 0,
            "only obj_1 exists (=1)": 0,
            "only obj_2 exists (>1)": 0,
            "only obj_2 exists (=1)": 0,
            "no target object": 0
        },
    }

    for idx, image_dir in tqdm(enumerate(glob.glob(os.path.join(image_path, "*.png"))), total=eval_num):
        # print("{}/{}".format(idx, eval_num))
        image_name = os.path.basename(image_dir)
        s_json, objects = {}, {}
        img = Image.open(os.path.join(image_path, image_name))

        first_half = image_name.split(" and ")[0].lower().replace("a ", "").replace("A ", "").replace("an ", "")
        second_half = image_name.split(" and ")[1].split("_")[0].lower().replace("a ", "").replace("A ", "").replace("an ", "")
        img_save = Image.new('RGB', (img.width * 2, img.height))
        for half_i, half in enumerate([first_half, second_half]):
            texts = [[half]]

            inputs = processor(text=texts, images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            image_copy = img.copy()
            draw = ImageDraw.Draw(image_copy)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([img.size[::-1]])
            # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.25)
            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                x1, y1, x2, y2 = tuple(box)
                draw.rectangle(xy=((x1, y1), (x2, y2)), width=4, outline=colors(label, True))
                
                object_key = text[label]
                if object_key in objects:
                    # Key exists, increment the index based on the current length of the list
                    index = len(objects[object_key]) + 1
                else:
                    # Key does not exist, initialize the list with index 1
                    objects[object_key] = []
                    index = 1
                objects[object_key].append(["#{}".format(index), box, round(score.item(), 3)])
                
                # draw label
                label_text = object_key + "#{}  ".format(index) + str(round(score.item(), 3))
                font = ImageFont.load_default(size=15)
                left, top, right, bottom = draw.textbbox((0, 0), label_text, font)
                w, h = right-left, bottom-top
                outside = box[1] - h >= 0  # label fits outside box
                draw.rectangle(
                            (box[0], box[1] - h if outside else box[1], 
                            box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1),
                            fill=colors(label, True),
                        )
                draw.text((box[0], box[1] - h if outside else box[1]), text=label_text, fill=(255, 255, 255), font=font)
                
            img_save.paste(image_copy, (half_i*img.width, 0))

        s_json["objects"] = objects
        res = check_mixture(s_json, first_half, second_half)
        s_json["summery"] = res
        summary['identity_counts'][res["identity"]] += 1

        img_save.save(os.path.join(save_path, img_path, exp_name, image_name.split(".")[0] + ".png"))
        with open(os.path.join(save_path, label_path, exp_name, "{}.json".format(image_name.split(".")[0])), 'w') as outfile:
            json.dump(s_json, outfile)
        
        result_list[image_name.split(".")[0]] = res
    
    with open(os.path.join(save_path, label_path, f"{exp_name}_all.json"), 'w') as outfile:
        json.dump(result_list, outfile)
    
    with open(os.path.join(save_path, label_path, f"{exp_name}_summary.json"), 'w') as outfile:
        json.dump(summary, outfile)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time/eval_num:.6f} seconds")
    return summary
