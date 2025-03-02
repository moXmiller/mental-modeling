print("started")
from datetime import datetime
print(datetime.now())

import os
import csv

from mm_utils import retrieve_inputs, extract_floats
from mm_text_additional_models import run_mllama_additional_models

print("everything but vllm loaded")
print(datetime.now())

from vllm import LLM, SamplingParams 
from vllm.utils import FlexibleArgumentParser

print("vllm imported")
print(datetime.now())

model_map_path = {
    "r1-distill-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
}

def inference_direct_text_only_small(llm, sampling_params, body, questions, args):
    inputs = []
    
    for idx, q_list in enumerate(questions):
        q = q_list[0]
        b = body[idx]
        
        prompt = b + " " + q
        
        inputs.append(prompt)

        print(f"inputs created")
                
    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    return outputs

def main(args):
    model = args.model_name
    depth = args.depth
    problem_type = args.problem_type
    sample_size = args.sample_size
    tensor_parallel_size = args.tensor_parallel_size

    model_path = model_map_path[model]

    body, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)
    
    llm = run_mllama_additional_models(tensor_parallel_size, model_path, model_name=model)
    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=2048,
                                    stop = [])
    
    outs = inference_direct_text_only_small(llm, sampling_params, body, questions, args)

    csv_field_names = ["problem_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]

    outs_list = []

    for idx, o in enumerate(outs):
        prompt = o.prompt
        out = o.outputs[0].text
        pred, gt = extract_floats(out, answers, idx)
        outputs = {"problem_idx": idx, "prompt": prompt, "generated_output": out, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
        outs_list.append(outputs)

    target_dir = f"data/mm/mental/{model}/cot"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_path = f"data/mm/mental/{model}/cot/outs_cot_{problem_type}_{depth}.csv"

    with open(target_path, 'w', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        csvwriter.writeheader()
        csvwriter.writerows(outs_list)

    print(f"Successfully written .csv file for depth {depth}!")

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="llama3.2-3b")
    parser.add_argument("--depth", "-d", type=int, default=1)
    parser.add_argument("--problem_type", type=str, default="trans")
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument("--tensor_parallel_size", '-tps', type=int, default=1)
    parser.add_argument("--time_generate", type=bool, default=True)
    args = parser.parse_args()
    main(args)