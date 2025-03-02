print("started")
from datetime import datetime
print(datetime.now())

import os
import csv
import re
import json
import pandas as pd
from transformers.image_utils import load_image

from mm_store_graphs import multimodal_prompt, retrieve_image_path, store_images
from mm_structured_mwps import partial_msp
from mm_utils import extract_floats_vlm, retrieve_inputs, extract_floats, construct_sentences
from mm_extract_correct_json import write_intermediate_json

print("everything but vllm and mm_mental_models loaded")

from mm_mental_models import complete_temp

print("everything but vllm loaded")
print(datetime.now())

from vllm import LLM, SamplingParams 
from vllm.utils import FlexibleArgumentParser

print("vllm imported")
print(datetime.now())


def run_mllama90(modality: str, tensor_parallel_size: int):
    llm = run_mllama(modality, tensor_parallel_size, model_name = "unsloth/Llama-3.2-90B-Vision-Instruct")
    return llm

def run_mllama(modality: str, tensor_parallel_size: int, model_name = "unsloth/Llama-3.2-11B-Vision-Instruct", download_dir = None):
    assert modality == "image"
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        swap_space=10,
        download_dir=download_dir
    )

    return llm

def get_multi_modal_input(image_path, question, args, step_by_step = False):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """

    if args.modality == "image":
        images = retrieve_images(image_path)
        for image in images:
            image = image.convert("RGB")
            if not step_by_step: img_question = multimodal_prompt(question)
            else: img_question = question
        
        return {
            "data": image,
            "question": img_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)

def get_multi_modal_input_white_box(problem, args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """

    if args.modality == "image":
        image_list = retrieve_images(images_path = None)
        image = image_list[0].convert("RGB")
        img_question = problem
        
        return {
            "data": image,
            "question": img_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)

model_map = {
    "llama3.2-11b": run_mllama,
    "llama3.2-90b": run_mllama90
}

def retrieve_images(images_path):
    images = []

    empty_image_path = "path/to/empty_image.mv.png" # set path to empty image
    if images_path == None: images_path = empty_image_path

    if not isinstance(images_path, list):
        images_path =  [images_path]

    if len (images_path)>0:
        for image_path in images_path:
            image = load_image(image_path)
            images.append(image)
            
    return images

def compose_mental_prompt_vlm(sentences, state_idx, new_prompt, lf_dict = {}):
    if state_idx == 0:
        ins_beginning = f"Extract structured information from the following sentence and update an empty JSON object with the categories: 'agent', 'quantity', 'entity', 'attribute', and 'unit'. If a category is not applicable, leave it as an empty string. Do not introduce additional categories." # We call this the JSON container representation." # \nTo help you, we provide:\n1.) A guide explaining how to construct and fill JSON container state logical forms.\n2.) {ex_pl} showing the step-by-step process.\nFollow the provided structure when creating JSON container representations." # \nWhen creating container representations based on the image, adhere strictly to the provided structure. Note that the agent's name ([LABEL]) is displayed at the top of the image. Use this name exactly as presented but ensure it is capitalized and formatted by removing any additional elements such as underscores or numbers. Replace [LABEL] with this cleaned and capitalized name only."
        ins = ins_beginning
        sentence = sentences[state_idx]
        sentence_ins = "This is the sentence: '" + sentence + "'."
        prompt = ins + "\n" + sentence_ins

    elif state_idx > 0: 
        lf = json.dumps(lf_dict)
        if new_prompt: ins_after = "Update a JSON representation based on a graphical difference relationship.\nYou are given:\n1.) A JSON object containing the agent's current state.\n2.) An image illustrating the relationship between the current and future quantity. Both quantities are represented as '?'.\nBased on this information, you should:\nA.) Extract the full JSON object.\nB.) Infer the future quantity using the information from the image.\nC.) Return the updated JSON object, ensuring that the 'quantity' field contains the final numerical value, not an equation or expression." 
        else: ins_after = "Update a JSON representation based on a graphical difference relationship.\nYou are given:\n1.) A JSON object containing the agent's current quantity.\n2.) An image illustrating the relationship between the current and future quantity. Both quantities are represented as '?'.\nBased on this information, you should:\nA.) Extract the current quantity from the JSON object.\nB.) Infer the future quantity using the information from the image.\nC.) Return an updated JSON object containing only the quantity, expressed as a number (not a term or equation)." # f"Interpret a graphical difference relationship and update a JSON representation accordingly. You are given:\n1.) A JSON object representing the agent's current quantity.\n2.) An image depicting the relationship between the current and future quantity.\n\nThe current and future quantities are both represented as '?'.\nFirst, deduce the current quantity from the JSON representation.\nThen, infer the future quantity based on the image.\nFinally, return the updated JSON object with the correct future quantity.\nState the quantity as a number, not as a term or equation. Only include the future quantity in the JSON representation."
        ins = ins_after
        lf_ins = "This is the JSON object: '" + lf + "'."
        prompt = ins + "\n" + lf_ins
    return prompt

def prepare_csv(questions, answers, outputs):
    outs_list = []

    global_count = 0

    print("started to read outputs")

    for idx, o in enumerate(outputs):
        answer = answers[idx][0]
        prompt = o.prompt
        generated_text = o.outputs[0].text
        pred, gt = extract_floats_vlm(generated_text, answer)
        if pred == gt: global_count += 1
        outputs = {"problem_idx": idx, "prompt": prompt, "generated_output": generated_text, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
        outs_list.append(outputs)
    print("global accuracy: ", global_count / len(questions))

    return outs_list

def write_csv(csv_field_names, model, problem_type, depth, outs_list, step_by_step=False, text_only=False, new_prompt = False, raw_outputs = False):
    dir_path = f"data/mm/mental/{model}/vlm_outputs"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if step_by_step and (not new_prompt): 
        if raw_outputs: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_raw.csv"
        else: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}.csv"
    elif step_by_step and new_prompt: 
        if raw_outputs: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_new_raw.csv"
        else: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_new.csv"
    elif text_only: 
        if raw_outputs: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_text_only_{problem_type}_{depth}_raw.csv"
        else: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_text_only_{problem_type}_{depth}.csv"
    else: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_{problem_type}_{depth}.csv"
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader()
        writer.writerows(outs_list)
    print("Successfully written .csv file!")

def write_csv_vlm_mental(csv_field_names, model, problem_type, depth, outs_list, sentence_idx, step_by_step=True, text_only=False, new_prompt = True, raw_outputs = True):
    dir_path = f"data/mm/mental/{model}/vlm_outputs"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if step_by_step and new_prompt and (not text_only) and raw_outputs: 
        csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_new_sentence_{sentence_idx}.csv"
    else: raise NotImplementedError("write_csv_vlm_mental() is not implemented")
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader()
        writer.writerows(outs_list)
    print("Successfully written .csv file!")

    if raw_outputs: return csv_file_path

def body_inference_vlm(body, sentence_idx: int, logical_forms: list, llm, sampling_params, args):
    problem_type, depth, modality, step_by_step, new_prompt = args.problem_type, args.depth, args.modality, args.step_by_step, args.new_prompt
    
    inputs = []
    for index, b in enumerate(body):
        sentences = construct_sentences(b)
        if sentence_idx > 0: image_path = retrieve_image_path(problem_type, depth, index, sentence_idx)
        prompt_text = compose_mental_prompt_vlm(sentences=sentences, state_idx=sentence_idx, new_prompt=new_prompt, lf_dict=logical_forms[index])
        if index == 0: print(f"prompt at sentence_idx {sentence_idx}: {prompt_text}")
        if sentence_idx == 0: mm_input = get_multi_modal_input_white_box(prompt_text, args)
        else: mm_input = get_multi_modal_input(image_path, prompt_text, args, step_by_step)
        data = mm_input["data"]
        question = mm_input["question"]

        prompt = f"<|image|><|begin_of_text|>{question}"

        input = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        }
        
        inputs.append(input)
        
    assert args.num_prompts > 0
    if args.image_repeat_prob is not None:
        raise NotImplementedError("apply_image_repeat() is not implemented")
    
    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
            
    return outputs

def within_accuracy_json(json = {}, msp = None, state_idx = 0):
    final_key = list(msp.states[state_idx].containers.keys())[-1]
    ans = msp.states[state_idx].containers[final_key].get_value()
    if isinstance(json, dict) and "quantity" in json.keys(): 
        json_to_check = json["quantity"]
        print(f"json_to_check: {json_to_check}")
        json_str = str(json_to_check)
        pred, gt = extract_floats(json_str, ans, idx = None)
    elif isinstance(json, set) and len(json) == 1:
        pred_str = str(json.pop())
        pred, gt = extract_floats(pred_str, ans, idx = None)
    elif isinstance(json, str):
        pred, gt = extract_floats(json, ans, idx = None)
    else:
        pred, gt = "-1", float(ans)
        print(f"json is not a dict, str, or set of length 1 at state index {state_idx}: {json} of type {type(json)}")
    return gt, pred

import re

def remove_operators(json_dict):
    if '"x +' in json_dict: return json_dict
    elif '"x -' in json_dict: return json_dict
    elif '"x *' in json_dict: return json_dict
    elif '"x /' in json_dict: return json_dict
    elif '+ x"' in json_dict: return json_dict
    elif '- x"' in json_dict: return json_dict
    elif '* x"' in json_dict: return json_dict
    elif '/ x"' in json_dict: return json_dict
    elif '"x+' in json_dict: return json_dict
    elif '"x-' in json_dict: return json_dict
    elif '"x*' in json_dict: return json_dict
    elif '"x/' in json_dict: return json_dict
    elif '+x"' in json_dict: return json_dict
    elif '-x"' in json_dict: return json_dict
    elif '*x"' in json_dict: return json_dict
    elif '/x"' in json_dict: return json_dict
    elif '"quantity +' in json_dict: return json_dict
    elif '"quantity -' in json_dict: return json_dict
    elif '_quantity +' in json_dict: return json_dict
    elif '_quantity -' in json_dict: return json_dict
    elif '"current quantity +' in json_dict: return json_dict
    elif '"current quantity -' in json_dict: return json_dict
    else:
        json_dict = re.sub(r'x\s*\+\s*(\d+)', r'"x + \1"', json_dict)
        json_dict = re.sub(r'x\s*\-\s*(\d+)', r'"x - \1"', json_dict)
        json_dict = re.sub(r'x\s*\*\s*(\d+)', r'"x * \1"', json_dict)
        json_dict = re.sub(r'x\s*\/\s*(\d+)', r'"x / \1"', json_dict)
        json_dict = re.sub(r'(\d+)\s*\+\s*x', r'"\1 + x"', json_dict)
        json_dict = re.sub(r'(\d+)\s*\-\s*x', r'"\1 - x"', json_dict)
        json_dict = re.sub(r'(\d+)\s*\*\s*x', r'"\1 * x"', json_dict)
        json_dict = re.sub(r'(\d+)\s*\/\s*x', r'"\1 / x"', json_dict)
        json_dict = re.sub(r'quantity\s*\+\s*(\d+)', r'"quantity + \1"', json_dict)
        json_dict = re.sub(r'quantity\s*\-\s*(\d+)', r'"quantity - \1"', json_dict)
    return json_dict

def replace_curly(text: str):
    text = re.sub(r'\{\\\{', '{', text)
    text = re.sub(r'\\\{', '{', text)
    text = re.sub(r'\{\{\{\{', '{', text)
    text = re.sub(r'\{\{\{', '{', text)
    text = re.sub(r'\{\{', '{', text)
    text = re.sub(r'\\\}', '}', text)
    text = re.sub(r'\s*\?\s*,', ' "?",', text) # replace question marks with "?"
    text = re.sub(r':\s*\?\s*\}', ': "?"}', text) # replace "quantity": ? with "quantity": "?"
    text = re.sub(r'\\\"', '"', text) # replace \" with "
    text = re.sub(r'\"future_quantity\"', 'future_quantity', text) # remove quotes around future_quantity
    text = re.sub(r'future_quantity', '"future_quantity"', text) # add quotes around future_quantity
    text = re.sub(r'\"<number>\"', '<number>', text) # remove quotes around <number>
    text = re.sub(r'<number>', '"<number>"', text) # add quotes around <number>
    text = re.sub(r'\"None\"', 'None', text) # remove quotes around None
    text = re.sub(r'None', '"None"', text) # add quotes around None
    text = re.sub(r':\sx\}', ': "x"}', text) # add quotes around x
    text = re.sub(r':\sy\}', ': "y"}', text) # add quotes around y
    text = re.sub(r':\sy\,', ': "y",', text) # add quotes around y
    text = re.sub(r'\"(\d+)\"', r'\1', text) # remove quotes around numbers
    text = re.sub(r'\"\-(\d+)\"', r'-\1', text) # remove quotes around negative numbers
    return text

def remove_empty_dict(text: str):
    new_forms = []
    old_forms = text.split("}")
    for f in old_forms:
        f = f.strip()
        if f not in [r"{",""]:
            f = f + "}"
            new_forms.append(f)
    if new_forms != []:
        if r"\{" in new_forms[-1]: new_json = new_forms[-1] # [0]
        else: new_json = new_forms[0]
        new_json = new_json[new_json.find('{'):]
        return new_json
    else: return text

def single_opening(text, index, sentence_idx):
    if len(re.findall(r"\{", text)) > 1: 
        print(f"Multiple opening braces detected at index {index}, sentence {sentence_idx}.")
        number_opening = len(re.findall(r"\{", text))
        text = text.replace("{", "", number_opening-1)
        text = text[text.find('{'):]
        return text
    else: return text

def write_json_lfs(lf_lfs, problem_type, model, depth, new_prompt = False, intermediate = False, sentence_idx = 0):
    lfs_dir = f"data/mm/mental/{model}/vlm_outputs"
    if not os.path.exists(lfs_dir):
        os.makedirs(lfs_dir)
    if intermediate: 
        if new_prompt: lfs_path = f"{lfs_dir}/logical_forms_json_{problem_type}_{depth}_sentence_{sentence_idx}_new.jsonl"
        else: lfs_path = f"{lfs_dir}/logical_forms_json_{problem_type}_{depth}_sentence_{sentence_idx}.jsonl"
    elif new_prompt: lfs_path = f"{lfs_dir}/logical_forms_json_{problem_type}_{depth}_new.jsonl"
    else: lfs_path = f"{lfs_dir}/logical_forms_json_{problem_type}_{depth}.jsonl"

    with open(lfs_path, 'w') as f:
        for lf_dict in lf_lfs:
            json.dump(lf_dict, f)
            f.write('\n')

    if intermediate: print(f"Created intermediate .jsonl for depth {depth} at sentence {sentence_idx}!")
    else: print(f"Created final .jsonl for depth {depth} and model {model}!")

def main(args, tensor_parallel_size = 4, template_file = "data/temp_no_plural.csv"):
    model = args.model_name
    depth = args.depth
    problem_type = args.problem_type
    sample_size = args.sample_size
    step_by_step = args.step_by_step
    text_only = args.text_only
    new_prompt = args.new_prompt
    raw_outputs = args.raw_outputs

    modality = args.modality

    stop_words = ["}"] + [f"$\\boxed{{{integer}}}$" for integer in range(-200, 201)] + ["I hope it is correct.",".0.0.0.0.0.0"] # ["**Answer:**", "####", "###", "*Answer:*","*Answer*","I hope it is correct."]
    body, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)
    
    size = len(body)

    llm = model_map[model](modality, tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=2048,
                                    stop = stop_words,
                                    include_stop_str_in_output = True)
    
    if step_by_step and (model == "llama3.2-90b"):
        stop_words_0 = ["The final answer is: "]
        sampling_params_0 = SamplingParams(temperature=0,
                                           max_tokens=2048,
                                           stop = stop_words_0,
                                           include_stop_str_in_output = True)
        stop_words_1 = []
        sampling_params_1 = SamplingParams(temperature=0,
                                           max_tokens=512, #2048,
                                           stop = stop_words_1,
                                           include_stop_str_in_output = True)
    
    
    if step_by_step:
        outs_list = []
        temp = complete_temp(problem_type)
        lf_lfs = [{} for _ in range(size)]
        print("started body inference")
        if model == "llama3.2-90b": ### trivial
            sentence_idx = 6
            if raw_outputs:
                if sentence_idx == 0: 
                    outs_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_new.csv"
                    outs = pd.read_csv(outs_path)
                    outs = outs[outs["sentence_idx"]==sentence_idx]
                else: 
                    lf_lfs = [{} for _ in range(size)]
                    prev_idx = sentence_idx - 1
                    if sentence_idx == 1: lfs_path = f"data/mm/mental/{model}/logical_forms_vision/logical_forms_json_{problem_type}_{depth}_sentence_{prev_idx}_new.jsonl"
                    elif sentence_idx == 6: lfs_path = f"data/mm/mental/{model}/logical_forms_vision/logical_forms_json_{problem_type}_{depth}_sentence_5_new.jsonl"
                    else: lfs_path = new_lfs_path
                    with open(lfs_path, 'r') as f:
                        for idx, line in enumerate(f):
                            lf_dict = json.loads(line)
                            lf_lfs[idx] = lf_dict
                    outs = body_inference_vlm(body, sentence_idx, lf_lfs, llm, sampling_params_1, args)
                print(f"finished body inference at sentence index {sentence_idx}")

                if sentence_idx == 0:
                    for index, o in enumerate(outs["generated_output"]):
                        prompt = outs["prompt"].iloc[index]
                        out = o # outs["generated_output"].iloc[index]
                        outputs = {"problem_idx": index, "sentence_idx": sentence_idx, "prompt": prompt, "generated_output": out}
                        outs_list.append(outputs)

                else:
                    for index, o in enumerate(outs):
                        prompt = o.prompt
                        out = o.outputs[0].text
                        outputs = {"problem_idx": index, "sentence_idx": sentence_idx, "prompt": prompt, "generated_output": out}
                        outs_list.append(outputs)

                csv_field_names = ["problem_idx", "sentence_idx", "prompt", "generated_output"]
                new_outs_path = write_csv_vlm_mental(csv_field_names, model, problem_type, depth, outs_list, sentence_idx, step_by_step, new_prompt=new_prompt, raw_outputs=True)

                new_lfs_path = write_intermediate_json(model, sentence_idx, depth, outs_path=new_outs_path, problem_type = problem_type, size = sample_size, text=False, image=True)

            else:
                if model == "llama3.2-90b":
                    if sentence_idx == 0: outs = body_inference_vlm(body, sentence_idx, lf_lfs, llm, sampling_params_0, args)
                    else: outs = body_inference_vlm(body, sentence_idx, lf_lfs, llm, sampling_params_1, args)
                else: outs = body_inference_vlm(body, sentence_idx, lf_lfs, llm, sampling_params, args)
                print(f"finished body inference at sentence index {sentence_idx}")
                for index, o in enumerate(outs):
                    parsed_json = lf_lfs[index]
                    msp = partial_msp(body[index], questions[index][0], index, problem_type, temp, depth)
                    prompt = o.prompt
                    out = o.outputs[0].text
                    if (model == "llama3.2-90b") and ("'" in out):
                        p = re.compile('(?<!\\\\)\'')
                        out = p.sub('\"', out)
                    if sentence_idx >= 0: print(f"output at index {index}: {out}") ### print output
                    if model == "llama3.2-90b": json_str = out[out.find('{'):]
                    elif new_prompt: json_str = out[out.find('{'):]
                    else: json_str = out[out.find('{"'):]
                    if model == "llama3.2-90b": json_str = remove_empty_dict(json_str)
                    json_str = replace_curly(json_str)
                    json_str = single_opening(json_str, index, sentence_idx)
                    if len(json_str) > 1 and (('"' in json_str) or ("'" in json_str)):
                        extracted_json = json_str
                        extracted_json = remove_operators(extracted_json)
                        if "'" in json_str:
                            p = re.compile('(?<!\\\\)\'')
                            extracted_json = p.sub('\"', extracted_json)
                        print(f"extracted_json at index {index}, sentence index {sentence_idx}: {extracted_json}")
                        parsed_json = json.loads(extracted_json)
                        
                    elif len(json_str) > 1:
                        parsed_json = json_str
                    
                    if (not isinstance(parsed_json, dict)) and (not isinstance(parsed_json, set)) and (not isinstance(parsed_json, str)): 
                        print(parsed_json)
                        print(type(parsed_json))
                        parsed_json = {parsed_json}
                    lf_lfs[index] = parsed_json

                    if parsed_json == {}: print(f"parsed_json is empty at problem index {index}, sentence {sentence_idx}")

                    if sentence_idx < depth:
                        write_json_lfs(lf_lfs, problem_type, model, depth, new_prompt = new_prompt, intermediate = True, sentence_idx = sentence_idx)

                    gt, pred = within_accuracy_json(parsed_json, msp, sentence_idx)
                    if sentence_idx == depth: assert gt == answers[index][0]
                    if pred == gt: print(f"accurate intermediate prediction at problem index {index}, sentence {sentence_idx}")
                    outputs = {"problem_idx": index, "sentence_idx": sentence_idx, "prompt": prompt, "generated_output": out, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
                    outs_list.append(outputs)

        if not raw_outputs: write_json_lfs(lf_lfs, problem_type, model, depth, new_prompt = new_prompt)
        if not raw_outputs: 
            csv_field_names = ["problem_idx", "sentence_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
            write_csv(csv_field_names, model, problem_type, depth, outs_list, step_by_step, new_prompt=new_prompt, raw_outputs=raw_outputs)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--depth', '-d', type=int, required=True, help="depth argument")
    parser.add_argument("--problem_type", "-p", type=str, default="trans",
                        choices=["add","trans","part","nonlinear"], 
                        help="problem_type argument")    
    parser.add_argument("--sample_size", type=int, default=400,
                        help="sample_size argument")    
    parser.add_argument('--model_name',
                        '-m',
                        type=str,
                        default="llama3.2-11b",
                        choices=model_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--step_by_step", "-sbs", type=int, 
                        default=0, choices = [0,1],
                        help="step_by_step argument")    
    parser.add_argument("--text_only", "-txt", type=int, 
                        default=0, choices = [0,1],
                        help="text_only argument")
    parser.add_argument('--new_prompt', '-newp', type=int, 
                        default=0, choices = [0,1], 
                        help="new_prompt argument")
    parser.add_argument('--raw_outputs', '-raw', type=int, 
                        default=0, choices = [0,1], 
                        help="raw_outputs argument")
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')

    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=None,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
        ' (if enabled)')

    parser.add_argument(
        '--mm-cache-preprocessor',
        action='store_true',
        help='If True, enable caching of multi-modal preprocessor/mapper.')

    parser.add_argument(
        '--time-generate',
        default=True,
        action='store_true',
        help='If True, then print the total generate() call time')
    

    args = parser.parse_args()

    main(args, template_file = "data/temp_no_plural.csv")