import re
import pandas as pd

def retrieve_model_ids(model_name):
    if model_name == "llama3-8b":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "llama3-70b":
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif model_name == "mixtral-8x7b":
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    elif model_name == "qwen2.5-0.5b":
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    elif model_name == "qwen2.5-3b":
        model_id = "Qwen/Qwen2.5-3B-Instruct"
    elif model_name == "qwen2.5-7b":
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif model_name == "qwen2.5-32b":
        model_id = "Qwen/Qwen2.5-32B-Instruct"
    elif model_name == "qwen2.5-70b":
        model_id = "Qwen/Qwen2.5-70B-Instruct"
    elif model_name == "deepseek-r1":
        model_id = "deepseek-ai/DeepSeek-R1"
    elif model_name == "deepseek-v3":
        model_id = "deepseek-ai/DeepSeek-V3"
    return model_id

def depth_interval(problem_type, mental = False):
    if not mental:
        if problem_type == "nonlinear":
            min_depth = 3
            max_depth = 6
        else:
            min_depth = 6
            max_depth = 10
    else: min_depth, max_depth = 1, 6
    return min_depth, max_depth

def retrieve_inputs(depth, problem_type, sample_size=400, random=True):
    if problem_type == "nonlinear":
        file_path = f'data/mm/mwps/nonlinear_mwps_{depth}.jsonl'
    else:
        file_path = f"data/mm/mwps/linear_mwps_{depth}_{problem_type}.jsonl"
    
    problems = pd.read_json(file_path, lines=True)
    
    if random == True:
        subset_problems = problems.sample(n=sample_size, random_state=9502)
        b, q, a = subset_problems["body"], subset_problems["questions"], subset_problems["answers"]
    else:
        b, q, a = problems["body"], problems["questions"], problems["answers"]
        if len(b) > sample_size:
            b, q, a = b[:sample_size], q[:sample_size], a[:sample_size]

    return b, q, a

def few_shot_examples(problem_type):
    if problem_type == "nonlinear": few_shot = 5
    else: few_shot = 12
    return few_shot

def extract_floats(response, answer, idx):
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?(?:,\d+)*', response)
    if nums == []: pred_str = "-1"
    else: pred_str = nums[-1]
    pred_str = str(pred_str)
    prediction = float(pred_str.replace(",",""))
    if idx == None: ground_truth = float(answer)
    else: ground_truth = float(answer.iloc[idx][0])
    return prediction, ground_truth

def extract_floats_vlm(output, answer):
    matches = re.findall(r"\$\boxed\{(\d+)\}\$", output)
    if len(matches) == 0:
        match_repeat = re.findall(r"-?\d+.0.0.0.0", output)
        if len(match_repeat) > 0:
            mr = match_repeat[0].split(".")[0]
            prediction, ground_truth = extract_floats(mr, answer, None)
        else: prediction, ground_truth = extract_floats(output, answer, None)
    else:
        values = [str(match) for match in matches]
        set_values = set(values)
        if len(set(values)) != 1: print(f"Multiple values detected in the output: {set_values}")
        values = values[0]
        prediction = float(values.replace(",",""))
        ground_truth = float(answer)
    return prediction, ground_truth

def construct_sentences(body):
    sentences = body.split(". ")
    final_sentence = sentences[-1]
    sentences = [s + "." for s in sentences[:-1]]
    sentences.append(final_sentence)
    return sentences