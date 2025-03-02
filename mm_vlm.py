print("started")
from datetime import datetime
print(datetime.now())

import os
import csv
from tqdm import tqdm
from transformers.image_utils import load_image

from mm_store_graphs import multimodal_prompt, retrieve_image_path, store_images
from mm_utils import extract_floats_vlm, retrieve_inputs

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

def get_multi_modal_input(image_path, question, args):
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
            img_question = multimodal_prompt(question)
        
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

    if not isinstance(images_path, list):
        images_path =  [images_path]

    if len (images_path)>0:
        for image_path in tqdm(images_path):
            image = load_image(image_path)
            images.append(image)
            
    return images

def main(args, tensor_parallel_size = 4, template_file = "data/temp_no_plural.csv"):
    model = args.model_name
    depth = args.depth
    problem_type = args.problem_type
    sample_size = args.sample_size

    modality = args.modality

    stop_words = ["**Answer:**", "####", "###", "*Answer:*","*Answer*","I hope it is correct."]


    if model == "llama3.2-11b": questions, answers, _ = store_images(depth, problem_type, sample_size, template_file)
    else: _, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)
    
    global_count = 0

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

    llm = model_map[model](modality, tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=2048,
                                    stop = stop_words)
        
    assert args.num_prompts > 0
    if args.image_repeat_prob is not None:
        raise NotImplementedError("apply_image_repeat() is not implemented")
    
    else:
        inputs = inputs

    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    csv_field_names = ["problem_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
    outs_list = []

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

    dir_path = f"data/mm/mental/{model}/vlm_outputs"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    csv_file_path = f"data/mm/mental/{model}/vlm_outputs/outs_{problem_type}_{depth}.csv"
    
    with open(csv_file_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(outs_list)
    print("Successfully written .csv file!")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--depth', type=int, required=True, help="depth argument")
    parser.add_argument("--problem_type", type=str, default="trans",
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