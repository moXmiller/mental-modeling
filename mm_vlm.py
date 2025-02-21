# import random
print("started")
from datetime import datetime
print(datetime.now())
#from vllm import LLM, SamplingParams

import os
import csv
from tqdm import tqdm
# import argparse
from transformers.image_utils import load_image
# from IPython.display import display

from mm_store_graphs import multimodal_prompt, retrieve_image_path, store_images
from mm_utils import extract_floats_vlm, retrieve_inputs

print("everything but vllm loaded")
print(datetime.now())

from vllm import LLM, SamplingParams
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm.assets.image import ImageAsset
# from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
print("vllm imported")
print(datetime.now())


def run_mllama90(modality: str, tensor_parallel_size: int):
    llm = run_mllama(modality, tensor_parallel_size, model_name = "unsloth/Llama-3.2-90B-Vision-Instruct")
    return llm

def run_mllama(modality: str, tensor_parallel_size: int, model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"):
    assert modality == "image"

    #meta-llama

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        swap_space=10,
        ### change model to $SCRATCH
        download_dir="/cluster/scratch/millerm/.cache/huggingface/hub"# ,
        #args.mm_cache_preprocessor,
    )

    # prompt = f"<|image|><|begin_of_text|>{question}"
    # stop_token_ids = None
    return llm #, prompt

def get_multi_modal_input(image_path, question, args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    #mathworld_repo/output_files/viz/asdiv-0002.gv.png
    #How many balls does Ellen have?

    if args.modality == "image":
        # Input image and question
        # images = retrieve_images("mathworld_repo/output_files/viz/mwp.gv.png")
        images = retrieve_images(image_path)
        for image in images:
            image = image.convert("RGB")
            # img_question = "You are tasked with interpreting a graphical representation of a difference relationship. To do so, answer the question below. You are not given the answer explicitly, but you should infer it from the graphical representation.\nBased on this information, answer the question.\nThis is the question: How many toasters does Janaye have?"
            img_question = multimodal_prompt(question)
        
        return {
            "data": image,
            "question": img_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


# def apply_image_repeat(image_repeat_prob, num_prompts, data, prompt, modality):
#     """Repeats images with provided probability of "image_repeat_prob". 
#     Used to simulate hit/miss for the MM preprocessor cache.
#     """
#     assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)
#     no_yes = [0, 1]
#     probs = [1.0 - image_repeat_prob, image_repeat_prob]

#     inputs = []
#     cur_image = data
#     for i in range(num_prompts):
#         if image_repeat_prob is not None:
#             res = random.choices(no_yes, probs)[0]
#             if res == 0:
#                 # No repeat => Modify one pixel
#                 cur_image = cur_image.copy()
#                 new_val = (i // 256 // 256, i // 256, i % 256)
#                 cur_image.putpixel((0, 0), new_val)

#         inputs.append({
#             "prompt": prompt,
#             "multi_modal_data": {
#                 modality: cur_image
#             }
#         })

#     return inputs

model_map = {
    "llama3.2-11b": run_mllama,
    "llama3.2-90b": run_mllama90
    # model_name = "unsloth/Llama-3.2-90B-Vision-Instruct"
}

def retrieve_images(images_path):
    images = []

    if not isinstance(images_path, list):
        images_path =  [images_path]

    # print(f"images_path: {images_path}")
    if len (images_path)>0:
        for image_path in tqdm(images_path):
            image = load_image(image_path)
            images.append(image)
            
    return images

# def chat_with_mllm (model, processor, prompt, images_path=[], max_new_tokens=4096, messages=[], images=[]):

#     # Ensure list:
#     if not isinstance(images_path, list):
#         images_path =  [images_path]

#     # Load images 
#     if len (images)==0 and len (images_path)>0:
#             for image_path in tqdm(images_path):
#                 image = load_image(image_path)
#                 images.append(image)
                
#     # If starting a new conversation about an image
#     if len (messages)==0:
#         messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

#     # # If continuing conversation on the image
#     # else:
#     #     messages.append ({"role": "user", "content": [{"type": "text", "text": prompt}]})

#     # process input data
#     text = processor.apply_chat_template(messages, add_generation_prompt=True)
#     inputs = processor(images=images, text=text, return_tensors="pt", ).to(model.device)

#     # Generate response
#     generation_args = {"max_new_tokens": max_new_tokens, "do_sample": False}
#     # if do_sample:
#     #     generation_args["temperature"] = temperature
#     generate_ids = model.generate(**inputs,**generation_args)
#     generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:-1]
#     generated_texts = processor.decode(generate_ids[0], clean_up_tokenization_spaces=False)

#     # Append the model's response to the conversation history
#     messages.append ({"role": "assistant", "content": [  {"type": "text", "text": generated_texts}]})

#     return generated_texts, messages, images

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
        # for state_idx in range(1, depth + 1):
        image_path = retrieve_image_path(problem_type, depth, idx)
        mm_input = get_multi_modal_input(image_path, q, args)
        data = mm_input["data"]
        # print(f"data: {data}")
        question = mm_input["question"]
        # print(f"question: {question}")
        # intermediate_answer = intermediate_answers[state_idx - 1]

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
        
        # intermediate_answers = intermediate_dict[idx]

        #############

    assert args.num_prompts > 0
    # if args.num_prompts == 1:
    #     inputs = {
    #         "prompt": prompt,
    #         "multi_modal_data": {
    #             modality: data
    #         },
    #     }

    # else:
        # Batch inference
    if args.image_repeat_prob is not None:
        # Repeat images with specified probability of "image_repeat_prob"
        # inputs = apply_image_repeat(args.image_repeat_prob,
        #                             args.num_prompts, data, prompt,
        #                             modality)
        raise NotImplementedError("apply_image_repeat() is not implemented")
    
    else:
        # inputs = [{
        #     "prompt": prompt,
        #     "multi_modal_data": {
        #         modality: data
        #     },
        # } for _ in range(args.num_prompts)]
        inputs = inputs

    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    # print("length of output: ", len(outputs))

    csv_field_names = ["problem_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
    outs_list = []

    print("started to read outputs")

    # for idx, q_list in enumerate(questions):
    #     answer = answers.iloc[idx][0]
    for idx, o in tqdm(enumerate(outputs)):
        # print(f"o: {o}")
        answer = answers[idx][0]
        # print(answer)
        prompt = o.prompt
        generated_text = o.outputs[0].text
        # print(f"generated_text: {generated_text}")
        # pred, gt = extract_floats_vlm(generated_text, intermediate_answer)
        pred, gt = extract_floats_vlm(generated_text, answer)
        # print(f"prediction: {pred}")
        # print(f"ground truth: {gt}")
    # if pred == gt: local_count += 1
    # if state_idx == depth - 1:
    #     pred, gt = extract_floats_vlm(generated_text, answer)
        if pred == gt: global_count += 1
    # print(f"prediction: {pred}")
    # print(f"ground truth: {gt}")
    # print(f"local accuracy at question {idx}: ", local_count / depth)
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