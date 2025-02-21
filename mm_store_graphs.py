from mathworld_repo.mathworld.utils.viz_helper import visualize_mwp_state

import os
import argparse
from mm_structured_mwps import create_re_lf, create_msp, sub_msp
from mm_text_to_lf import create_re_temp, load_temp_table
from mm_utils import retrieve_inputs
from tqdm import tqdm

print("Imported modules successfully.")

def store_images(depth, problem_type, sample_size, template_file = "data/temp_no_plural.csv", step_by_step=False):
    template_file = 'data/temp_no_plural.csv'
    temp = load_temp_table(template_file)
    temp = create_re_temp(temp)
    temp = create_re_lf(temp)

    dir_path = f"/cluster/scratch/millerm/images_mw/{problem_type}/depth_{depth}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    body, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)

    size = len(body)
    intermediate_dict = {}

    assert size == sample_size, f"size: {size} and sample_size: {sample_size}"

    # if sample_size == 400: print(f"sample_size at 400: no additional graphs created")
    # else:
    #     body = body.reset_index(drop=True)
    #     questions = questions.reset_index(drop=True)
    #     answers = answers.reset_index(drop=True)
    #     print("all indices reset")

    if step_by_step:

        for idx in tqdm(range(0,size)):
            sub, im_ans = sub_msp(temp, body[idx], questions[idx][0], idx, problem_type, depth)

            for state_idx in range(0,len(sub.states)-1):
                mwp_name = f"mwp_{depth}_{idx}_{state_idx}"
                visualize_mwp_state(sub.states[state_idx], mwp_name, path = dir_path, sub_idx = state_idx)
            intermediate_dict[idx] = im_ans

    # mwp_name = f"mwp_{depth}_{idx}"
    # # print(sub.states)
    # # print(sub.states[depth].containers)
    # visualize_mwp_state(sub.states[depth], mwp_name, path = dir_path)
    
    elif not step_by_step:
        msps_list = create_msp(temp, depth, problem_type, sample_size)

        for msp_index, msp in enumerate(msps_list):
            mwp_name = f"mwp_{depth}_{msp_index}"
            visualize_mwp_state(msp.final, mwp_name, path = dir_path)

    # questions = questions.reset_index(drop=True)
    # answers = answers.reset_index(drop=True)

    return questions, answers, intermediate_dict

def retrieve_image_path(problem_type, depth, idx, state_idx = None):
    dir_path = f"/cluster/scratch/millerm/images_mw/{problem_type}/depth_{depth}"
    if state_idx == None: mwp_name = f"mwp_{depth}_{idx}" # _{state_idx}"
    else: mwp_name = f"mwp_{depth}_{idx}_{state_idx}"
    return f"{dir_path}/{mwp_name}.gv.png"
    # return f"mathworld_repo/output_files/viz/mwp_{state_idx}.gv.png"

def multimodal_prompt(question):
    instruction = "You are tasked with interpreting a graphical representation of a difference relationship. To do so, answer the question below. You are not given the answer explicitly, but you should infer it from the graphical representation.\nBased on this information, answer the following question. Reason step by step."
    prompt = instruction + "\n" + "Q: " + question + "\n" + "A: "
    prompt = str(prompt)
    return prompt

def store_empty_images(problem_type = "trans", depth = 3, sample_size = 400, template_file = 'data/temp_no_plural.csv'):
    temp = load_temp_table(template_file)
    temp = create_re_temp(temp)
    temp = create_re_lf(temp)

    empty_dir_path = f"/cluster/scratch/millerm/images_mw/{problem_type}"
    if not os.path.exists(empty_dir_path):
        os.makedirs(empty_dir_path)

    body, questions, _ = retrieve_inputs(depth, problem_type, sample_size, random=False)

    sub, _ = sub_msp(temp, body[0], questions[0][0], 0, problem_type, depth)

    sub_idx = depth + 1
    mwp_name = f"mwp_empty"
    visualize_mwp_state(sub.states[sub_idx], mwp_name, path = empty_dir_path, sub_idx = sub_idx)
    return mwp_name

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Slurm job with problem_type and depth arguments")
    # parser.add_argument("--problem_type", default="trans", type=str, choices=["add","trans","part","nonlinear"], help="problem_type argument")
    # parser.add_argument("--depth", '-d', type=int, required=True, help="depth argument")
    # parser.add_argument("--sample_size", default = 400, type=int, help="sample_size argument")
    # parser.add_argument("--step_by_step", '-sbs', default = 0, type=int, choices = [0,1], help="step_by_step argument")
    # args = parser.parse_args()

    # problem_type = args.problem_type
    # depth = args.depth
    # sample_size = args.sample_size
    # step_by_step = args.step_by_step
    # step_by_step = bool(step_by_step)

    # template_file = "data/temp_no_plural.csv"
    # questions, answers, intermediate_dict = store_images(depth, problem_type, sample_size, template_file, step_by_step=step_by_step)
    
    store_empty_images()