print("started")

from tqdm import tqdm
from random import choice
from collections import Counter
import re
import csv
import os
import argparse

print("packages imported")

from mm_utils import retrieve_inputs, extract_floats, retrieve_model_ids, construct_sentences
from mm_text_to_lf import load_temp_table, load_json, map_concepts_to_path, extract_unique_concepts, unit_existence, read_lines_from_path, create_re_temp, fill_logical_form
from mm_structured_mwps import create_re_lf, partial_msp

print("cross-dependencies imported")

from vllm import LLM, SamplingParams

print("vllm imported")

def initialize_vllm(model_name: str, download_dir = None):    
    model_id = retrieve_model_ids(model_name)

    if model_name in ["llama3-70b", "mixtral-8x7b", "qwen2.5-70b","deepseek-r1","deepseek-v3"]: 
        llm = LLM(model=model_id,
                  download_dir=download_dir,
                  tensor_parallel_size=4)
    else: llm = LLM(model=model_id,
                    download_dir=download_dir)                
    print(f"{model_name} initialized successfully!")
    return llm

def generate_vllm(llm, prompts: str, max_tokens=4096):
  """
  Generate completions from the LLM.
  :param llm: Model instance.
  :param prompt: Text of the prompt.
  :param num_samples: Number of samples to generate.
  :param temperature: Temperature for sampling.
  :return: List of generated texts, List of the raw outputs
  """
  sampling_params = SamplingParams(temperature=0, n=1, max_tokens=max_tokens)
  outputs = llm.generate(prompts, sampling_params, use_tqdm = True)
  return outputs

def get_container_lfs(temp):
    temp = temp[temp["type"] == "container"]
    return temp.reset_index()

def retrieve_agent_name(sentence, template_file = 'data/temp_no_plural.csv'):
    temp = load_temp_table(template_file)
    temp = get_container_lfs(temp)
    temp = create_re_temp(temp)
    for id, tmp in enumerate(temp["re_temp"]):
        label_match = re.match(tmp,sentence)
        if label_match:
            return label_match.group(id+1)

def retrieve_context_lfs(ent_unit_path = "data/ent_unit_names.json", template_file = 'data/temp_no_plural.csv'):
    temp = load_temp_table(template_file)
    temp = get_container_lfs(temp)
    concepts = extract_unique_concepts(temp)
    ent_unit_names = load_json(ent_unit_path)
    concepts_to_path, entity_lfs, unit_lfs, entity_to_unit = map_concepts_to_path(concepts, part_whole = False)
    temp_id = 0
    concept_pattern = r"\[([A-Z0-9_]+)\]"
    target_concepts = re.findall(concept_pattern, temp["temp"][temp_id])        
    target_concepts = [f"[{concept}]" for concept in target_concepts]
    unit_exists = unit_existence(target_concepts, unit_lfs)
    empty_lf = temp["lf"][temp_id]
    lf = temp["lf"][temp_id]
    text = temp["temp"][temp_id]
    for concept in concepts:
        if concept in text: 
            line = f"[{concept}]"
            bracket_concept = f"[{concept}]"
            if bracket_concept in unit_lfs: 
                pass
            elif (bracket_concept in entity_lfs) and (unit_exists):
                line = choice(list(ent_unit_names))
                unit_lf = entity_to_unit[bracket_concept]
                unit = ent_unit_names[line]
                text = text.replace(unit_lf, unit)
                lf = lf.replace(unit_lf, unit + "s")
            elif concept != "NUM":
                target_path = concepts_to_path[concept]
                lines = read_lines_from_path(target_path)
                line = choice(lines)
            else:
                line = str(choice(range(2,20)))
            text = text.replace(f"[{concept}]", line)
            lf = lf.replace(f"[{concept}]", line)
    return empty_lf, lf, text

def compose_prompt(text, initial_sentence, lf_list = [], ic_total = 2):
    total_lfs = len(lf_list)
    if ic_total != 1: ex_pl = f"{ic_total} examples"
    else: ex_pl = f"{ic_total} example"
    if text == initial_sentence:
        ins_beginning = f"Translate the given natural language sentence into a container representation. To help you, we provide:\n1.) A guide explaining how to construct and fill container state logical forms.\n2.) {ex_pl} showing the step-by-step process.\nFollow the provided structure when creating container representations."
    if text != initial_sentence: 
        agent = retrieve_agent_name(initial_sentence)
        provided_containers = f"the provided container representation for agent {agent}"
        if lf_list != []: lf = lf_list[0]
        else: lf = ""
    elif total_lfs > 1: 
        provided_containers = f"the provided container representation for agent {agent}" # f"one or several of the {total_lfs} provided container representations"
        lf = "; ".join(lf_list)
    else: lf = None
    if text != initial_sentence: ins_after = f"Translate the natural language sentence into a container representation. To do so, modify {provided_containers} using additional information from the sentence.\nFocus only on 'container' representations. This means:\nA.) We are only interested in the final quantity of entities that a single agent possesses.\nB.) Each representation should reflect a single agent's state in terms of their relationship with the entities.\nC.) Even if the sentence suggests a different type of representation, trace the argument until it updates the relevant agent's container state to reflect the total amount they have."
    if text == initial_sentence: ins = ins_beginning
    else: ins = ins_after
    text_ins = "This is the natural language sentence to process: '" + text + "'."
    if text != initial_sentence: lf_ins = "This is the container representation: '" + lf + "'."
    context_ins = f"Below, we provide you with the {ex_pl} in natural language together with the corresponding empty and filled container representations."
    ic_list = []
    ic_examples = ""
    for i in range(0,ic_total):
        context_empty_lf, context_filled_lf, context_text = retrieve_context_lfs()
        ic_example = "Natural language sentence: '" + context_text + "', empty container representation: '" + context_empty_lf + "', filled container representation: '" + context_filled_lf + "'."
        ic_list.append(context_filled_lf)
        if i < ic_total - 1: ic_examples = ic_examples + ic_example + "\n"
        else: ic_examples = ic_examples + ic_example
    if text == initial_sentence: prompt = ins + "\n" + text_ins + "\n" + context_ins + "\n" + ic_examples
    else: prompt = ins + "\n" + text_ins + "\n" + lf_ins
    return prompt, ic_list

def compose_question_prompt(question, lf_list: list):
    if lf_list != []: lf = "; ".join(lf_list)
    else: lf = ""
    ins = "Answer the question using the container representations."
    q_ins = "This is the question to answer: '" + question + "'."
    lf_ins = "This is the container representation: '" + lf + "'."
    q_prompt = ins + "\n" + q_ins + "\n" + lf_ins
    return q_prompt

def multiple_lf_list(lf_string):
    opening_patterns = re.findall(r"\s\(\s", lf_string)
    if len(opening_patterns) > 1:
        lf_repl = lf_string.replace(" ),", " ),,")
        lf_list = lf_repl.split(",,")
        return lf_list
    else: return [lf_string]

def retrieve_containers(output, prev_cont = [], ic_cont = [], vlm = False):
    cont_omit = prev_cont + ic_cont
    if vlm: pattern = r"container\s?\(\s?([\w]+[A-Za-z0-9\s\d\,\-]+)\s?\)"
    else: pattern = r"container\s?\(\s?([A-Z]{1}[a-z]+[A-Za-z0-9\s\d\,\-]+)\s?\)"
    if output != None: 
        ap = re.findall(pattern, output)
        ap = [p for p in ap if "None" in p]
    else: ap = []
    fp = []
    for i, p in enumerate(ap):
        p = f"container ( {p})"
        ap[i] = p
        if p not in cont_omit:
            fp.append(p)
    if cont_omit != []: return fp
    else: return ap

def most_common_containers(containers, sentence_idx = 1):
    if containers != []:
        occ = Counter(containers)
        mc = occ.most_common(sentence_idx)[0][1]
        l = [o[0] for o in occ.most_common() if o[1]>=mc]
        return l
    else: return containers

def within_accuracy(container = [], msp = None, state_idx = 0):
    final_key = list(msp.states[state_idx].containers.keys())[-1]
    ans = msp.states[state_idx].containers[final_key].get_value()
    if container != []: 
        cont_to_check = container[0]
        pred, gt = extract_floats(cont_to_check, ans, idx = None)
    else: 
        pred = float(-1)
        gt = float(ans)
    return gt, pred

def body_inference(body, sentence_idx: int, logical_forms: list, llm):
    prompts = []
    ic_lists = []
    for index, b in enumerate(tqdm(body)):
        text = construct_sentences(b)[sentence_idx]
        initial_sentence = construct_sentences(b)[0]
        prompt, ic_list = compose_prompt(text, initial_sentence, logical_forms[index])
        prompts.append(prompt)
        ic_lists.append(ic_list)        
    outputs = generate_vllm(llm, prompts)
    return outputs, ic_lists

def question_inference(questions, logical_forms: list, llm):
    question_prompts = []
    for index, question in tqdm(enumerate(questions)):
        q = question[0]
        question_prompt = compose_question_prompt(q, logical_forms[index])
        question_prompts.append(question_prompt)
    question_outputs = generate_vllm(llm, question_prompts)
    return question_outputs

def complete_temp(problem_type, template_file = 'data/temp_no_plural.csv'):
    if problem_type != "part":
        temp = load_temp_table(template_file)
        temp = create_re_temp(temp)
        temp = create_re_lf(temp)
    else: raise NotImplementedError
    return temp

def compute_mental_accuracy(depth = 6, problem_type = "part", model_name = "llama3-8b", sample_size = 400):
    llm = initialize_vllm(model_name)
    body, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)
    size = len(body)
    csv_field_names = ["problem_idx", "sentence_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]
    outs_list = []
    
    temp = complete_temp(problem_type)

    lf_lfs = [[] for _ in range(size)]

    print("started body inference")
    for sentence_idx in range(0,depth + 1):
        outs, ic_containers = body_inference(body, sentence_idx, lf_lfs, llm)
        print("finished body inference")
        for index, o in enumerate(outs):
            lfs = lf_lfs[index]
            msp = partial_msp(body[index], questions[index][0], index, problem_type, temp, depth)
            prompt = o.prompt
            out = o.outputs[0].text
            conts = retrieve_containers(out, lfs, ic_containers)
            if conts != []:
                lfs = choice(most_common_containers(conts))
                lfs = [lfs]
                lf_lfs[index] = lfs
            gt, pred = within_accuracy(lfs, msp, sentence_idx)
            if pred == gt: print(f"accurate intermediate prediction at problem index {index}, sentence {sentence_idx}")
            outputs = {"problem_idx": index, "sentence_idx": sentence_idx, "prompt": prompt, "generated_output": out, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
            outs_list.append(outputs)

    count = 0
    print("started question inference")
    questions_outs = question_inference(questions, lf_lfs, llm)
    q_sentence_idx = "q"
    print("finished question inference")
    for index, o in enumerate(questions_outs):
        q_prompt = o.prompt
        out = o.outputs[0].text
        if isinstance(out, str): pred, gt = extract_floats(str(out), answers, index)
        else: print(f"generated_output not str: {out}\n\ntype: {type(out)}\n\ninstance str: {isinstance(out, str)}")
        if pred == gt:
            count += 1
        print(f"count at index {index}: {count}")
        outputs = {"problem_idx": index, "sentence_idx": q_sentence_idx, "prompt": q_prompt, "generated_output": out, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
        outs_list.append(outputs)
        
    acc = count / size
    print(f"accuracy for {problem_type}: {acc}")
    
    dir_path = f"data/mm/mental/{model_name}/outputs"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    csv_file_path = f"data/mm/mental/{model_name}/outputs/outs_{problem_type}_{depth}.csv"
        
    with open(csv_file_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(outs_list)
    print("Successfully written .csv file!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Slurm job with problem_type and depth arguments")
    parser.add_argument("--model_name", type=str, choices=["llama3-8b","llama3-70b","mixtral-8x7b","qwen2.5-0.5b","qwen2.5-3b","qwen2.5-7b","qwen2.5-32b","qwen2.5-70b","deepseek-r1","deepseek-v3","phi-3.5"], required=True, help="model_name argument")
    parser.add_argument("--problem_type", type=str, default = "trans", choices=["add","trans","part","nonlinear"], help="problem_type argument")
    parser.add_argument("--depth", type=int, required=True, help="depth argument")
    parser.add_argument("--sample_size", type=int, default = 400, help="sample_size argument")
    args = parser.parse_args()

    model_name = args.model_name
    problem_type = args.problem_type
    depth = args.depth
    sample_size = args.sample_size

    compute_mental_accuracy(depth = depth, problem_type= problem_type, model_name= model_name, sample_size=sample_size)