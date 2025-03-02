print("started")
from datetime import datetime
print(datetime.now())

import os
import csv
import json

from mm_utils import retrieve_inputs, construct_sentences
from mm_extract_correct_json import write_intermediate_json

print("everything but vllm loaded")
print(datetime.now())

from vllm import LLM, SamplingParams 
from vllm.utils import FlexibleArgumentParser

print("vllm imported")
print(datetime.now())


def run_mllama_additional_models(tensor_parallel_size: int, model_path = None, download_dir = None, model_name = None):
    if model_path == None: llm = LLM(
        model=model_map_path[model_name],
        max_model_len=4096,
        max_num_seqs=16,
        tensor_parallel_size=tensor_parallel_size,
        swap_space=10,
        trust_remote_code = True,
        download_dir=download_dir
    )
    elif download_dir == None: llm = LLM(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=16,
        tensor_parallel_size=tensor_parallel_size,
        swap_space=10,
        trust_remote_code = True
    )

    return llm

model_map_path = {
    "r1-distill-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
}

def compose_mental_prompt_text(sentences, state_idx, lf_dict = {}):
    if state_idx == 0:
        ins_beginning = f"Extract structured information from the following sentence and update an empty JSON object with the categories: 'agent', 'quantity', 'entity', 'attribute', and 'unit'. If a category is not applicable, leave it as an empty string. Do not introduce additional categories." # We call this the JSON container representation." # \nTo help you, we provide:\n1.) A guide explaining how to construct and fill JSON container state logical forms.\n2.) {ex_pl} showing the step-by-step process.\nFollow the provided structure when creating JSON container representations." # \nWhen creating container representations based on the image, adhere strictly to the provided structure. Note that the agent's name ([LABEL]) is displayed at the top of the image. Use this name exactly as presented but ensure it is capitalized and formatted by removing any additional elements such as underscores or numbers. Replace [LABEL] with this cleaned and capitalized name only."
        ins = ins_beginning
        sentence = sentences[state_idx]
        sentence_ins = "This is the sentence: '" + sentence + "'."
        prompt = ins + "\n" + sentence_ins

    elif state_idx > 0: 
        lf = json.dumps(lf_dict)
        ins_after = "Update the given JSON representation based on a described quantity change.\nYou are given:\n1.) A JSON object representing the agent's current state.\n2.) A natural language sentence that describes a change in quantity for the agent.\nBased on this information, you should:\nA.) Extract the full JSON object.\nB.) Interpret the sentence to determine how the quantity changes.\nC.) Return the updated JSON object, ensuring that the 'quantity' field contains the final numerical value, not an equation or expression."
        ins = ins_after
        sentence_ins = "This is the natural language sentence: '" + sentences[state_idx] + "'."
        lf_ins = "This is the JSON object: '" + lf + "'."
        prompt = ins + "\n" + sentence_ins + "\n" + lf_ins
    
    return prompt

def write_csv_text_mental(csv_field_names, model, problem_type, depth, outs_list, step_by_step=False, text_only=False, new_prompt = False, raw_outputs = False, sentence_idx = 0):
    dir_path = f"data/mm/mental/{model}/text_mental_modeling"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if step_by_step and (not new_prompt) and (not text_only): 
        if raw_outputs: csv_file_path = f"{dir_path}/outs_sbs_json_{problem_type}_{depth}_sentence_{sentence_idx}_text.csv"
        else: csv_file_path = f"{dir_path}/outs_sbs_json_{problem_type}_{depth}.csv"
    elif step_by_step and new_prompt and (not text_only): 
        if raw_outputs: csv_file_path = f"{dir_path}/outs_sbs_json_{problem_type}_{depth}_new_sentence_{sentence_idx}_text.csv"
        else: csv_file_path = f"{dir_path}/outs_sbs_json_{problem_type}_{depth}_new.csv"
    elif step_by_step and text_only:
        if raw_outputs: csv_file_path = f"{dir_path}/outs_sbs_text_only_{problem_type}_{depth}_sentence_{sentence_idx}_text.csv"
        else: csv_file_path = f"{dir_path}/outs_sbs_text_only_{problem_type}_{depth}.csv"
    elif text_only: 
        if raw_outputs: csv_file_path = f"{dir_path}/outs_text_only_{problem_type}_{depth}_sentence_{sentence_idx}_text.csv"
        else: csv_file_path = f"{dir_path}/outs_text_only_{problem_type}_{depth}.csv"
    else: csv_file_path = f"{dir_path}/outs_{problem_type}_{depth}.csv"
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader()
        writer.writerows(outs_list)
    print("Successfully written .csv file!")

    if raw_outputs: 
        print(csv_file_path)
        return csv_file_path

def body_inference_text(body, sentence_idx: int, logical_forms: list, llm, sampling_params, args):
    
    inputs = []
    for index, b in enumerate(body):
        sentences = construct_sentences(b)
        prompt_text = compose_mental_prompt_text(sentences=sentences, state_idx=sentence_idx, lf_dict=logical_forms[index])
        if index == 0: print(f"prompt at sentence_idx {sentence_idx}: {prompt_text}")
        
        input = prompt_text
        if index >= 390: print(f"index {index}")
        inputs.append(input)

    print("inputs generated")

    if args.time_generate:
        import time
        print("time imported")
        start_time = time.time()
        print("generate starting")
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        print("generate finished")
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        print("generate starting without time")
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        print("generate finished without time")
            
    return outputs

def main(args, template_file = "data/temp_no_plural.csv"):
    model = args.model_name
    depth = args.depth
    problem_type = args.problem_type
    sample_size = args.sample_size
    step_by_step = args.step_by_step
    text_only = args.text_only
    new_prompt = args.new_prompt
    tensor_parallel_size = args.tensor_parallel_size

    model_path = model_map_path[model]

    stop_words = []

    body, _, _ = retrieve_inputs(depth, problem_type, sample_size, random=False)
    
    size = len(body)

    llm = run_mllama_additional_models(tensor_parallel_size, model_path, model_name=model)
    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=768,
                                    stop = stop_words,
                                    include_stop_str_in_output = True)
    
    
    if step_by_step:
        outs_list = []
        print("started body inference")
        for sentence_idx in range(0,depth + 1):
            lf_lfs = [{} for _ in range(size)]
            if sentence_idx >= 1:
                lfs_path = new_lfs_path
                with open(lfs_path, 'r') as f:
                    for idx, line in enumerate(f):
                        lf_dict = json.loads(line)
                        lf_lfs[idx] = lf_dict
            outs = body_inference_text(body, sentence_idx, lf_lfs, llm, sampling_params, args)
            print(f"finished body inference at sentence index {sentence_idx}")
        
            for index, o in enumerate(outs):
                prompt = o.prompt
                out = o.outputs[0].text
                outputs = {"problem_idx": index, "sentence_idx": sentence_idx, "prompt": prompt, "generated_output": out}
                outs_list.append(outputs)

            csv_field_names = ["problem_idx", "sentence_idx", "prompt", "generated_output"]
            new_outs_path = write_csv_text_mental(csv_field_names, model, problem_type, depth, outs_list, step_by_step, new_prompt=new_prompt, raw_outputs=True)

            new_lfs_path = write_intermediate_json(model, sentence_idx, depth, outs_path=new_outs_path, problem_type = problem_type, size = sample_size, text = True, image = False)


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
    parser.add_argument("--tensor_parallel_size", "-tps", type=int, default=1,
                        help="tensor_parallel_size argument")    
    parser.add_argument("--model_name", "-m", type=str, 
                        # default="llama3.1-8b", 
                        choices=model_map_path.keys())
    
    parser.add_argument("--step_by_step", "-sbs", type=int, 
                        default=0, choices = [0,1],
                        help="step_by_step argument")    
    parser.add_argument("--text_only", "-txt", type=int, 
                        default=0, choices = [0,1],
                        help="text_only argument")
    parser.add_argument('--new_prompt', '-newp', type=int, 
                        default=0, choices = [0,1], 
                        help="new_prompt argument")
    
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    
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