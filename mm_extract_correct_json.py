import pandas as pd
import json
import re
import os

model = "llama3.2-90b"
depth = 1
problem_type = "trans"
size = 400
sentence_idx = 1
image = True
text = not image

def check_if_equation(ooo):
    all_equation_lists = re.findall(r"\{[\S\s]*agent[\S\s]*:[\S\s]*,[\S\s]*quantity[\S\s]*:[\S\s]*\d+[\S\s]*\d+[\S\s]*,[\S\s]*entity[\S\s]*attribute[\S\s]*unit[\S\s]*\"\s*\}", ooo)
    if all_equation_lists != []:
        ooo = all_equation_lists[-1]
        # print(ooo)
        if re.search(r'"\s*\d+\s*\+\s*\d+\s*"', ooo): return ooo
        if re.search(r'"\s*\d+\s*\-\s*\d+\s*"', ooo): return ooo
        if "-" in ooo: ooo = re.sub(r'(\d+\s*-\s*\d+)', r'"\1"', ooo)
        if "+" in ooo: ooo = re.sub(r'(\d+\s*+\s*\d+)', r'"\1"', ooo)        
    else:
        if '"updated_quantity"' in ooo: return ooo
        if 'updated_quantity' in ooo: ooo = re.sub(r'updated_quantity', '"updated_quantity"', ooo)
    return ooo

def write_intermediate_json(model, sentence_idx, depth, outs_path, problem_type = "trans", size = 400, text = False, image = False):
    lf_lfs = [{} for _ in range(size)]
    outs = pd.read_csv(outs_path)
    outs = outs[outs["sentence_idx"]==sentence_idx]
    generated_outputs = outs["generated_output"]
    for idx, o in enumerate(generated_outputs):
        if isinstance(o,str) and ("'" in o):
            p = re.compile('(?<!\\\\)\'')
            oo = p.sub('\"', o)
        else: oo = o
        if isinstance(oo,str): 
            json_objects = oo.split("Step 1")
            if len(json_objects) > 1:
                ooo = json_objects[1]
            else: ooo = oo
        else: ooo = oo
        if isinstance(ooo,str): 
            ooo = re.sub(r'\\boxed\{(\-*\d+)\}', '', ooo)
            if model == "llama3.2-3b": ooo = re.sub(r'\\', '', ooo)
            if "agent" in ooo and "quantity" in ooo and "entity" in ooo and "attribute" in ooo and "unit" in ooo and re.search(r'\s*\d+\s*', ooo):
                all_dicts = re.findall(r"\{\s*\"\s*agent\s*\"\s*:\s*\"*\s*[A-Za-z\-\_\s]*\s*\"*\s*,\s*\"\s*quantity\s*\"\s*:\s*\"*\s*\w*\s*\d+\s*\w*\s*\"*\s*,\s*\"\s*entity\s*\"\s*:\s*\"*\s*[A-Za-z\-\_\s]*\s*\"*\s*,\s*\"\s*attribute\s*\"\s*:\s*\"*\s*[A-Za-z\-\_\s]*\s*\"*\s*,\s*\"\s*unit\s*\"\s*:\s*\"*\s*[A-Za-z\-\_\d+\s]*\s*\"*\s*\}", ooo)
                
                if all_dicts != []:
                    json_str = all_dicts[-1]
                    json_str = json_str[:json_str.rfind('}')+1]
                    json_str = json_str[json_str.rfind('{'):]
                    json_str = json_str[:json_str.find('}')+1]
                    if ("Gael" in json_str) and ("22" in json_str) and ("rope" in json_str) and ("meters" in json_str) and (depth == 1) and (sentence_idx == 1) and (model == "llama3.2-3b"): json_str = json_str.replace('Gael', 'Gael"') # adaptation for llama3.2-3b, sentence_idx 1, depth 1
                    if 'Carleen"s keys' in json_str: json_str = json_str.replace('Carleen"s keys', 'Carleen\'s keys') # adaptation for llama3.2-3b, sentence_idx 0, depth 5, index 14
                    json_str = json_str.replace(": None", ': "None"')
                    json_str = check_if_equation(json_str)
                    if idx == 0: print(json_str)
                    lf_dict = json.loads(json_str)
                    lf_lfs[idx] = lf_dict
                else: 
                    print(f"No logical form at index {idx}")
                    
            else: 
                print(f"No logical form at index {idx}")
        else: 
            print(f"No logical form at index {idx}")

    if image == True: lfs_dir = f"data/mm/mental/{model}/logical_forms_vision"
    elif text: lfs_dir = f"data/mm/mental/{model}/text_mental_modeling"
    else: lfs_dir = f"data/mm/mental/{model}/vlm_outputs"
    if not os.path.exists(lfs_dir):
        os.makedirs(lfs_dir)
    if text: lfs_path = f"{lfs_dir}/logical_forms_json_{problem_type}_{depth}_sentence_{sentence_idx}_new_text.jsonl"
    else: lfs_path = f"{lfs_dir}/logical_forms_json_{problem_type}_{depth}_sentence_{sentence_idx}_new.jsonl"

    with open(lfs_path, 'w') as f:
        for lf_dict in lf_lfs:
            json.dump(lf_dict, f)
            f.write('\n')

    print(f"Created intermediate .jsonl for depth {depth} at sentence {sentence_idx}!")
    print(lfs_path)
    return lfs_path