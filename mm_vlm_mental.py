print("started")
from datetime import datetime
print(datetime.now())

import os
import csv
from tqdm import tqdm
from random import choice
from transformers.image_utils import load_image

from mm_store_graphs import multimodal_prompt, retrieve_image_path, store_images
from mm_structured_mwps import partial_msp
from mm_utils import extract_floats_vlm, retrieve_inputs

print("everything but vllm and mm_mental_models_fix loaded")

from mm_mental_models import retrieve_context_lfs, retrieve_containers, most_common_containers, within_accuracy, complete_temp

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
        for image_path in tqdm(images_path):
            image = load_image(image_path)
            images.append(image)
            
    return images

def compose_mental_prompt_vlm(state_idx, lf_list = [], ic_total = 2):
    if ic_total != 1: ex_pl = f"{ic_total} examples"
    else: ex_pl = f"{ic_total} example"
    if state_idx == 0:
        ins_beginning = f"Deduce the agent's current quantity from the image. You are provided with\n1.) A guide explaining how to construct and fill container state logical forms.\n2.) {ex_pl} showing the step-by-step process of creating a logical form from a natural language sentence.\nWhen creating container representations based on the image, adhere strictly to the provided structure. Note that the agent's name ([LABEL]) is displayed at the top of the image. Use this name exactly as presented but ensure it is capitalized and formatted by removing any additional elements such as underscores or numbers. Replace [LABEL] with this cleaned and capitalized name only."
        ins = ins_beginning
    elif state_idx > 0: 
        if lf_list != []: lf = lf_list[0]
        else: lf = ""
        ins_after = f"You are tasked with interpreting a graphical difference relationship. You are given:\n1.) A container representation showing the agent's current quantity.\n2.) An image depicting the relationship between the future quantity and the current quantity.\nUpdate the container representation based on the difference indicated in the image. Both the current and future quantities are represented as '?'. Hence, you must logically deduce the current quantity from the container representation first."
        ins = ins_after
        lf_ins = "This is the container representation: '" + lf + "'."
    context_ins = f"Below, we provide you with the {ex_pl} in natural language together with the corresponding empty and filled container representations."
    ic_list = []
    ic_examples = ""
    for i in range(0,ic_total):
        context_empty_lf, context_filled_lf, context_text = retrieve_context_lfs()
        ic_example = "Natural language sentence: '" + context_text + "', empty container representation: '" + context_empty_lf + "', filled container representation: '" + context_filled_lf + "'."
        ic_list.append(context_filled_lf)
        if i < ic_total - 1: ic_examples = ic_examples + ic_example + "\n"
        else: ic_examples = ic_examples + ic_example
    if state_idx == 0: prompt = ins + "\n" + context_ins + "\n" + ic_examples
    else: prompt = ins + "\n" + lf_ins
    return prompt, ic_list

def inference_direct(llm, sampling_params, questions, problem_type, depth, modality):
    inputs = []
    for idx, q_list in enumerate(questions):
        q = q_list[0]
        image_path = retrieve_image_path(problem_type, depth, idx)
        mm_input = get_multi_modal_input(image_path, q, args)
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

        print(f"inputs created")
                
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

def inference_direct_text_only(llm, sampling_params, body, questions, modality):
    inputs = []
    for idx, q_list in enumerate(questions):
        q = q_list[0]
        b = body[idx]
        problem_ins = b + " " + q
        print(problem_ins)
        mm_input = get_multi_modal_input_white_box(problem_ins, args)
        question = mm_input["question"]
    
        prompt = question
        
        input = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: None # data
            },
        }

        inputs.append(input)

        print(f"inputs created")
                
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

def prepare_csv(questions, answers, outputs):
    outs_list = []

    global_count = 0

    print("started to read outputs")

    for idx, o in tqdm(enumerate(outputs)):
        answer = answers[idx][0]
        prompt = o.prompt
        generated_text = o.outputs[0].text
        pred, gt = extract_floats_vlm(generated_text, answer)
        if pred == gt: global_count += 1
        outputs = {"problem_idx": idx, "prompt": prompt, "generated_output": generated_text, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
        outs_list.append(outputs)
    print("global accuracy: ", global_count / len(questions))

    return outs_list

def write_csv(csv_field_names, model, problem_type, depth, outs_list, step_by_step=False, text_only=False):
    dir_path = f"data/mm/mental/{model}/vlm_outputs"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if step_by_step: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_sbs_{problem_type}_{depth}.csv"
    elif text_only: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_text_only_{problem_type}_{depth}.csv"
    else: csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_{problem_type}_{depth}.csv"
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader()
        writer.writerows(outs_list)
    print("Successfully written .csv file!")

def body_inference_vlm(body, sentence_idx: int, logical_forms: list, llm, problem_type, depth, modality, sampling_params, step_by_step=True):
    
    inputs = []
    ic_lists = []
    for index, _ in tqdm(enumerate(body)):
        image_path = retrieve_image_path(problem_type, depth, index, sentence_idx)
        prompt_text, ic_list = compose_mental_prompt_vlm(state_idx=sentence_idx, lf_list=logical_forms[index])
        mm_input = get_multi_modal_input(image_path, prompt_text, args, step_by_step)
        
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
        ic_lists.append(ic_list)      

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
            
    return outputs, ic_lists

def main(args, tensor_parallel_size = 4, template_file = "data/temp_no_plural.csv"):
    model = args.model_name
    depth = args.depth
    problem_type = args.problem_type
    sample_size = args.sample_size
    step_by_step = args.step_by_step
    text_only = args.text_only

    modality = args.modality

    stop_words = [f"$\\boxed{{{integer}}}$" for integer in range(-200, 201)] + ["I hope it is correct.",".0.0.0.0.0.0"] # ["**Answer:**", "####", "###", "*Answer:*","*Answer*","I hope it is correct."]


    if (model == "llama3.2-11b") and (not step_by_step) and (not text_only): questions, answers, _ = store_images(depth, problem_type, sample_size, template_file, step_by_step) ### we should create the images separately!
    else: body, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)
    
    size = len(body)

    llm = model_map[model](modality, tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=2048,
                                    stop = stop_words,
                                    include_stop_str_in_output = True)
    
    if (not step_by_step) and (not text_only):
        outputs = inference_direct(llm, sampling_params, questions, problem_type, depth, modality, args)
        outs_list = prepare_csv(questions, answers, outputs)
        csv_field_names = ["problem_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
        write_csv(csv_field_names, model, problem_type, depth, outs_list)

    elif text_only:
        outputs = inference_direct_text_only(llm, sampling_params, body, questions, modality)
        outs_list = prepare_csv(questions, answers, outputs)
        csv_field_names = ["problem_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
        write_csv(csv_field_names, model, problem_type, depth, outs_list, text_only=True)

    else:
        outs_list = []
        temp = complete_temp(problem_type)
        lf_lfs = [[] for _ in range(size)]
        print("started body inference")
        for sentence_idx in range(0,depth + 1):
            outs, ic_containers = body_inference_vlm(body, sentence_idx, lf_lfs, llm, problem_type, depth, modality, sampling_params, step_by_step)
            print(f"finished body inference at sentence index {sentence_idx}")
            for index, o in enumerate(outs):
                lfs = lf_lfs[index]
                msp = partial_msp(body[index], questions[index][0], index, problem_type, temp, depth)
                prompt = o.prompt
                out = o.outputs[0].text
                conts = retrieve_containers(out, lfs, ic_containers, vlm=True)
                # print(f"index {index}: containers: {conts}, output text: {out}")
                if conts != []:
                    lfs = choice(most_common_containers(conts))
                    lfs = [lfs]
                    lf_lfs[index] = lfs
                gt, pred = within_accuracy(lfs, msp, sentence_idx)
                if pred == gt: print(f"accurate intermediate prediction at problem index {index}, sentence {sentence_idx}")
                outputs = {"problem_idx": index, "sentence_idx": sentence_idx, "prompt": prompt, "generated_output": out, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
                outs_list.append(outputs)

        csv_field_names = ["problem_idx", "sentence_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
        write_csv(csv_field_names, model, problem_type, depth, outs_list, step_by_step)


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