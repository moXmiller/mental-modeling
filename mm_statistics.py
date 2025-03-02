import pandas as pd
import csv
import numpy as np
import os
from random import choices
import json

from mm_utils import depth_interval, extract_floats

def write_statistics_csv(model_name, problem_type, random_context=False, num_bootstrap=100):
    if random_context == False: stats_path = f"data/mm/{model_name}/statistics/stats_{problem_type}.csv"
    else: stats_path = f"data/mm/{model_name}/statistics/stats_{problem_type}_random_context.csv"
    csv_field_names = ["model_name", "problem_type", "c_type", "depth", "accuracy", "lower_quantile", "upper_quantile"]
    c_types = [0, 1, 2, 3]
    min_depth, max_depth = depth_interval(problem_type)
    stats_list = []

    for c_type in c_types:
        if (problem_type == "nonlinear" and c_type == 3): max_depth = 5
        for depth in range(min_depth, max_depth+1):
            statistics = {}
            if random_context == False: outs_path = f"data/mm/{model_name}/outputs/outs_{problem_type}_t{c_type}_{depth}.csv"
            else: outs_path = f"data/mm/{model_name}/outputs/outs_{problem_type}_t{c_type}_{depth}_random_context.csv"
            outs = pd.read_csv(outs_path)
            accuracy = outs["accurate"].mean()

            bs = np.zeros(num_bootstrap)
            for i in range(0,num_bootstrap):
                bs_mean = compute_bootstrap(outs)
                bs[i] = bs_mean
            lower_quantile = np.quantile(bs, 0.025)
            upper_quantile = np.quantile(bs, 0.975)
            
            statistics = {"model_name": model_name, "problem_type": problem_type, "c_type": c_type, "depth": depth, "accuracy": accuracy, "lower_quantile": lower_quantile, "upper_quantile": upper_quantile}
            stats_list.append(statistics)

    with open(stats_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(stats_list)
    print(f"Successfully written .csv file for {problem_type} problems!")

def compute_bootstrap(df, mental = False, accuracies = None, sample_size = 400):
    if df is not None:
        if mental: df = df.reset_index()
        samples = choices(df["accurate"], k=len(df["accurate"]))

    if accuracies != None:
        samples = choices(accuracies, k=sample_size)
    
    bs_mean = np.array(samples).mean()
    return bs_mean

def manipulate_mental_outs(model_name, problem_type):
    min_depth, max_depth = depth_interval(problem_type, mental=True)
    updated_csv_field_names = ["problem_idx", "sentence_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]

    for depth in range(min_depth, max_depth+1):
        outs_path = f"data/mm/mental/{model_name}/outputs/outs_{problem_type}_{depth}.csv"
        upd_outs_path = f"data/mm/mental/{model_name}/outputs/outs_{problem_type}_{depth}_updated.csv"
        outs = pd.read_csv(outs_path)
        if "sentence_idx" not in list(outs.columns):
            sentences_indices = [i for i in range(depth + 1) for _ in range(400)]
            outs["sentence_idx"] = sentences_indices
            outs = outs[updated_csv_field_names]
            outs.to_csv(upd_outs_path)

def manipulate_cot_outs(model_name, problem_type = "trans"):
    min_depth, max_depth = depth_interval(problem_type, mental=True)
    updated_csv_field_names = ["problem_idx", "prompt", "generated_output", "prediction", "ground_truth", "accurate"]

    for depth in range(min_depth, max_depth+1):
        outs_path = f"data/mm/mental/{model_name}/cot/outs_cot_{problem_type}_{depth}.csv"
        upd_outs_path = f"data/mm/mental/{model_name}/cot/outs_cot_{problem_type}_{depth}.csv"
        outs = pd.read_csv(outs_path)
        outs_list = []
        for idx, o in enumerate(outs["generated_output"]):
            splits = o.split("Step 1")
            if len(splits) > 1:
                o = splits[1]
            gt = outs["ground_truth"].iloc[idx]
            pred, gt = extract_floats(o, gt, None)
            outputs = {"problem_idx": idx, "prompt": outs["prompt"].iloc[idx], "generated_output": o, "prediction": pred, "ground_truth": gt, "accurate": int(pred == gt)}
            outs_list.append(outputs)

            with open(upd_outs_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                csvwriter = csv.DictWriter(csvfile, fieldnames=updated_csv_field_names)
                csvwriter.writeheader()
                csvwriter.writerows(outs_list)

def add_to_statistics(reduced_outs, stats_list, accs_dict, intermediate_index, num_bootstrap, model_name, problem_type, depth):
    statistics = {}

    bs = np.zeros(num_bootstrap)
    for i in range(0,num_bootstrap):
        bs_mean = compute_bootstrap(reduced_outs, mental=True)
        bs[i] = bs_mean
    lower_quantile = np.quantile(bs, 0.025)
    upper_quantile = np.quantile(bs, 0.975)
    
    acc = accs_dict[intermediate_index]
    statistics = {"model_name": model_name, "problem_type": problem_type, "depth": depth, "intermediate_step": intermediate_index, "accuracy": acc, "lower_quantile": lower_quantile, "upper_quantile": upper_quantile}
    stats_list.append(statistics)
    
    return stats_list

def write_statistics_mental(model_name, problem_type, num_bootstrap=100):
    dir_path = f"data/mm/mental/{model_name}/statistics"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_sbs.csv"
    csv_field_names = ["model_name", "problem_type", "depth", "intermediate_step", "accuracy", "lower_quantile", "upper_quantile"]
    min_depth, max_depth = depth_interval(problem_type, mental=True)
    
    stats_list = []

    for depth in range(min_depth, max_depth):
        outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}.csv"
        outs = pd.read_csv(outs_path)
        
        accs = {}
        for intermediate_idx in range(0,depth+1):
            outs_int = outs[outs["sentence_idx"]==intermediate_idx]
            acc_int = outs_int["accurate"].mean()
            accs[intermediate_idx] = acc_int
            stats_list = add_to_statistics(outs_int, stats_list, accs, intermediate_idx, num_bootstrap, model_name, problem_type, depth)
        
    with open(stats_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(stats_list)
    print(f"Successfully written .csv file for model {model_name}!")

def write_statistics_vlm(model_name, problem_type, step_by_step, text_only, new_prompt = False, sbs_text = False, num_bootstrap = 100): # sbs_text = True if sbs and text_only
    dir_path = f"data/mm/mental/{model_name}/statistics"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if text_only: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_text_only.csv"
    elif step_by_step: 
        if new_prompt: 
            if sbs_text: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_sbs_new_text.csv"
            else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_vision_mental_modeling.csv"
        else: 
            if sbs_text: raise NotImplementedError("Not implemented yet!")
            else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_sbs.csv"
    else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}.csv"
    if not step_by_step: csv_field_names = ["model_name", "problem_type", "depth", "accuracy", "lower_quantile", "upper_quantile"]
    else: csv_field_names = ["model_name", "problem_type", "depth", "intermediate_step", "accuracy", "lower_quantile", "upper_quantile"]
    min_depth, max_depth = depth_interval(problem_type, mental=True)
    
    stats_list = []
    
    for depth in range(min_depth, max_depth+1):
        if text_only: outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_text_only_{problem_type}_{depth}.csv"
        elif step_by_step: 
            if new_prompt: 
                if sbs_text: outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_new.csv"
                else: outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_vision_mental_modeling.csv"
            else: outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}.csv"
        else: outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_{problem_type}_{depth}.csv"
        outs = pd.read_csv(outs_path)
        
        if step_by_step:
            accs = {}
            for intermediate_idx in range(0,depth+1):
                outs_int = outs[outs["sentence_idx"]==intermediate_idx]
                acc_int = outs_int["accurate"].mean()
                accs[intermediate_idx] = acc_int
                stats_list = add_to_statistics(outs_int, stats_list, accs, intermediate_idx, num_bootstrap, model_name, problem_type, depth)

        else:
            accuracy = outs["accurate"].mean()

            bs = np.zeros(num_bootstrap)
            for i in range(0,num_bootstrap):
                bs_mean = compute_bootstrap(outs)
                bs[i] = bs_mean
            lower_quantile = np.quantile(bs, 0.025)
            upper_quantile = np.quantile(bs, 0.975)
            
            statistics = {"model_name": model_name, "problem_type": problem_type, "depth": depth, "accuracy": accuracy, "lower_quantile": lower_quantile, "upper_quantile": upper_quantile}
            stats_list.append(statistics)

    with open(stats_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(stats_list)
    print(f"Successfully written .csv file for model {model_name}!")

def write_statistics_from_logical_forms(model_name, text = True, problem_type = "trans", num_bootstrap = None, sample_size = 400):
    image = not text

    stats_list = []

    if text: 
        stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_text_mental_modeling.csv"
        if num_bootstrap != None: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_text_mental_modeling_bs.csv"
    if image: 
        stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_vision_mental_modeling.csv"
        if num_bootstrap != None: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_vision_mental_modeling_bs.csv"

    stats_dir = f"data/mm/mental/{model_name}/statistics"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    for depth in range(1, 7):
        for sentence_idx in range(0, depth + 1):
            
            if text: 
                if (model_name == "llama3.2-90b") and (depth == 1): outs_path = f"data/mm/mental/{model_name}/text_mental_modeling/outs_sbs_json_trans_1_new_sentence_0_text.csv"
                elif model_name in ["llama3.2-3b", "llama3.3-70b", "r1-distill-qwen", "r1-distill-llama"]: outs_path = f"data/mm/mental/{model_name}/text_mental_modeling/outs_sbs_json_trans_{depth}_new_sentence_0_text.csv"
                else: outs_path = f"data/mm/mental/{model_name}/text_mental_modeling/outs_sbs_json_trans_{depth}_new_sentence_{sentence_idx}_text.csv"
            if image: outs_path = f"data/mm/mental/{model_name}/vlm_outputs/outs_sbs_json_{problem_type}_{depth}_new.csv"

            if text: lfs_path = f"data/mm/mental/{model_name}/text_mental_modeling/logical_forms_json_trans_{depth}_sentence_{sentence_idx}_new_text.jsonl"
            if image: lfs_path = f"data/mm/mental/{model_name}/logical_forms_vision/logical_forms_json_{problem_type}_{depth}_sentence_{sentence_idx}_new.jsonl"
            
            gt_outs_path = f"data/mm/mental/llama3.2-90b/vlm_outputs/outs_sbs_json_trans_{depth}.csv"
            
            lf_lfs = [{} for _ in range(sample_size)]
            with open(lfs_path, 'r') as f:
                for idx, line in enumerate(f):
                    lf_dict = json.loads(line)
                    lf_lfs[idx] = lf_dict

            outs = pd.read_csv(outs_path)
            gt_outs = pd.read_csv(gt_outs_path)

            outs = outs[outs["sentence_idx"] == sentence_idx]
            gt_outs = gt_outs[gt_outs["sentence_idx"] == sentence_idx]

            count = 0
            
            accuracies = [0] * 400
            for idx, lf in enumerate(lf_lfs):
                if lf != {}: 
                    quant_str = str(lf["quantity"])
                    o = outs["generated_output"].iloc[idx]
                    gt = gt_outs["ground_truth"].iloc[idx]
                    
                    pred, gt = extract_floats(quant_str, gt, None)
                    
                    if pred == gt:
                        count += 1
                        accuracies[idx] = 1
                    
                    elif sentence_idx == depth: 
                        pred, gt = extract_floats(o, gt, None)
                        if pred == gt: 
                            count += 1
                            accuracies[idx] = 1
            
            accuracy = count / sample_size

            if num_bootstrap != None: 
                bs = np.zeros(num_bootstrap)
                for i in range(0,num_bootstrap):
                    bs_mean = compute_bootstrap(None, accuracies=accuracies, sample_size=sample_size)
                    bs[i] = bs_mean
                lower_quantile = np.quantile(bs, 0.025)
                upper_quantile = np.quantile(bs, 0.975)
                
                statistics = {"model_name": model_name, "problem_type": problem_type, "depth": depth, "sentence_idx": sentence_idx, "accuracy": accuracy, "lower_quantile": lower_quantile, "upper_quantile": upper_quantile}
                csv_field_names = ["model_name", "problem_type", "depth", "sentence_idx", "accuracy", "lower_quantile", "upper_quantile"]

            else: 
                statistics = {"model_name": model_name, "problem_type": problem_type, "depth": depth, 
                              "sentence_idx": sentence_idx, "accuracy": accuracy}
                csv_field_names = ["model_name", "problem_type", "depth", "sentence_idx", "accuracy"]
            
            stats_list.append(statistics)

    with open(stats_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(stats_list)

    print(stats_path)
    print(f"Successfully written .csv file for model {model_name}!")
    
def write_statistics_cot(model_name, problem_type = "trans", num_bootstrap=100):
    dir_path = f"data/mm/mental/{model_name}/statistics"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_cot.csv"
    csv_field_names = ["model_name", "problem_type", "depth", "accuracy", "lower_quantile", "upper_quantile"]
    min_depth, max_depth = depth_interval(problem_type, mental=True)
    stats_list = []

    for depth in range(min_depth, max_depth+1):
        statistics = {}
        outs_path = f"data/mm/mental/{model_name}/cot/outs_cot_{problem_type}_{depth}.csv"
        
        outs = pd.read_csv(outs_path)
        accuracy = outs["accurate"].mean()

        bs = np.zeros(num_bootstrap)
        for i in range(0,num_bootstrap):
            bs_mean = compute_bootstrap(outs)
            bs[i] = bs_mean
        lower_quantile = np.quantile(bs, 0.025)
        upper_quantile = np.quantile(bs, 0.975)
        
        statistics = {"model_name": model_name, "problem_type": problem_type, "depth": depth, "accuracy": accuracy, "lower_quantile": lower_quantile, "upper_quantile": upper_quantile}
        stats_list.append(statistics)

    with open(stats_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(stats_list)
    print(stats_path)
    print(f"Successfully written .csv file for {problem_type} problems!")

def plot_statistics(model_name, problem_type, savefig = False, text_only = False, step_by_step = False, new_prompt = False, sbs_text = False):
    if text_only: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_text_only.csv"
    elif step_by_step: 
        if new_prompt: 
            if sbs_text: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_sbs_new_text.csv"
            # else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_sbs_new.csv"
            else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_vision_mental_modeling.csv"
        else:
            if sbs_text: raise NotImplementedError("Not implemented yet!") 
            else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}_sbs.csv"
    else: stats_path = f"data/mm/mental/{model_name}/statistics/stats_{problem_type}.csv"
    data = pd.read_csv(stats_path)
    import matplotlib.pyplot as plt
    import seaborn as sns

    depths = sorted(data['depth'].unique())
    
    # if model_name in ["llama3.2-11b","llama3.2-90b"]: vlm = True
    # else: vlm = False

    bar_width = 0.2

    if step_by_step: 
        steps = sorted(data['intermediate_step'].unique())
        group_width = len(steps) * bar_width + 0.1

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, depth in enumerate(depths):
        group_data = data[data['depth'] == depth]
        if step_by_step:
            for j, step in enumerate(steps):
                step_data = group_data[group_data['intermediate_step'] == step]
                
                if not step_data.empty:
                    bar_position = i * group_width + j * bar_width
                    ax.bar(
                        bar_position,
                        step_data['accuracy'].values[0],
                        width=bar_width,
                        yerr=[[step_data['accuracy'].values[0] - step_data['lower_quantile'].values[0]],
                            [step_data['upper_quantile'].values[0] - step_data['accuracy'].values[0]]],
                        capsize=5,
                        label=f"Step {step}" if i == 5 else "",
                        color=sns.color_palette("muted")[j % len(sns.color_palette("muted"))],
                        alpha=0.8
                    )

        else:
            bar_position = i * bar_width
            ax.bar(
                bar_position,
                group_data['accuracy'].values[0],
                width=bar_width,
                yerr=[[group_data['accuracy'].values[0] - group_data['lower_quantile'].values[0]],
                    [group_data['upper_quantile'].values[0] - group_data['accuracy'].values[0]]],
                capsize=5,
                label=f"Depth {depth}",
                color=sns.color_palette("muted")[i % len(sns.color_palette("muted"))],
                alpha=0.8
            )

    if not step_by_step: ax.set_xticks([i * bar_width for i in range(len(depths))])
    else: ax.set_xticks([i * group_width + (group_width - bar_width) / 2 for i in range(len(depths))])
    ax.set_xticklabels([f"Depth {depth}" for depth in depths])

    ax.set_xlabel("Depth", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    if sbs_text: ax.set_title(f"Accuracy by Depth for Model {model_name} for Text Mental Modeling", fontsize=14)
    elif step_by_step and new_prompt and not sbs_text: ax.set_title(f"Accuracy by Depth for Model {model_name} for Vision Mental Modeling", fontsize=14)
    else: ax.set_title(f"Accuracy by Depth for Model {model_name}", fontsize=14)
    if not step_by_step: ax.legend(title="Depth", fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    else: ax.legend(title="Intermediate Steps", fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()

    if savefig:
        if text_only: save_fig_path = f"data/mm/mental/{model_name}/statistics/results_{problem_type}_{model_name}_text_only.png"
        elif step_by_step: 
            if new_prompt: 
                if sbs_text: save_fig_path = f"data/mm/mental/{model_name}/statistics/results_{problem_type}_{model_name}_sbs_new_text.png"
                # else: save_fig_path = f"data/mm/mental/{model_name}/statistics/results_{problem_type}_{model_name}_sbs_new.png"
                else: save_fig_path = f"data/mm/mental/{model_name}/statistics/results_{problem_type}_{model_name}_vision_mental_modeling.png"
            else: 
                if sbs_text: raise NotImplementedError("Not implemented yet!")
                else: save_fig_path = f"data/mm/mental/{model_name}/statistics/results_{problem_type}_{model_name}_sbs.png"
        else: save_fig_path = f"data/mm/mental/{model_name}/statistics/results_{problem_type}_{model_name}.png"
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    else: plt.show()

def write_comparison_statistics(model_name, problem_type = "trans", new_prompt = False):
    stats_dir = f"data/mm/mental/{model_name}/statistics"
    if new_prompt: stats_files = [f"stats_{problem_type}{x}.csv" for x in ["_text_only", "", "_sbs_new"]]
    else: stats_files = [f"stats_{problem_type}{x}.csv" for x in ["_text_only", "", "_sbs"]]
    stats_list = []
    for f in stats_files:
        file = os.path.join(stats_dir, f)
        stats = pd.read_csv(file)
        if (f == f"stats_{problem_type}_sbs.csv") or (f == f"stats_{problem_type}_sbs_new.csv"): 
            stats = stats[stats["intermediate_step"]==stats["depth"]]
            technique = "mental modelling with new prompt" if new_prompt else "mental modelling"
        if f == f"stats_{problem_type}_text_only.csv": technique = "language only"
        if f == f"stats_{problem_type}.csv": technique = "vision only"
        depth, accuracy = stats["depth"].reset_index(drop=True), stats["accuracy"].reset_index(drop=True)
        for a_idx, a in enumerate(accuracy):
            statistics = {"method": technique, "model_name": model_name, "depth": depth[a_idx], "accuracy": a}
            stats_list.append(statistics)

    if new_prompt: final_path = stats_dir + f"/comparison_stats_{problem_type}_{model_name}_new.csv"
    else: final_path = stats_dir + f"/comparison_stats_{problem_type}_{model_name}.csv"
    csv_field_names = ["method", "model_name", "depth", "accuracy"]
    with open(final_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(stats_list)
    print(f"Successfully written .csv file for model {model_name}!")

def retrieve_superior_outputs(model_name, problem_type = "trans"): ### still need to modify for new prompt
    superior_outs = []
    for depth in range(1, 6 + 1):
        outs_dir = f"data/mm/mental/{model_name}/vlm_outputs"
        outs_files = [f"outs_{x}_{problem_type}_{depth}.csv" for x in ["text_only", "sbs_json"]]
        compare_dict = {"language only": [], "mental modelling": []}
        file_text = os.path.join(outs_dir, outs_files[0])
        outs_text = pd.read_csv(file_text)
        accs_text = outs_text["accurate"]
        file_sbs = os.path.join(outs_dir, outs_files[1])
        outs_sbs = pd.read_csv(file_sbs)
        outs_sbs = outs_sbs[outs_sbs["sentence_idx"]==depth].reset_index(drop=True)
        accs_sbs = outs_sbs["accurate"]
        compare_dict["language only"] = accs_text
        compare_dict["mental modelling"] = accs_sbs
        for idx in range(0, len(compare_dict["mental modelling"])):
            if compare_dict["mental modelling"][idx] > compare_dict["language only"][idx]:
                print(f"Superior output found at index {idx} for depth {depth} and model {model_name}!")
                prompt_text = outs_text["prompt"].iloc[idx]
                generated_output_text = outs_text["generated_output"].iloc[idx]
                prompt_mental = outs_sbs["prompt"].iloc[idx]
                generated_output_mental = outs_sbs["generated_output"].iloc[idx]
                compared_outs = {"depth": depth, "problem_idx": idx, "prompt_text": prompt_text, "generated_output_text": generated_output_text, "prompt_mental": prompt_mental, "generated_output_mental": generated_output_mental}
                superior_outs.append(compared_outs)

    stats_dir = f"data/mm/mental/{model_name}/statistics"
    superior_path = stats_dir + f"/comparison_outs_{problem_type}_{model_name}.csv"
    csv_field_names = ["depth", "problem_idx", "prompt_text", "generated_output_text", "prompt_mental", "generated_output_mental"]

    with open(superior_path, 'w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(superior_outs)
    print(f"Successfully written .csv file for model {model_name}!")

if __name__ == "__main__":
    write_statistics_cot("r1-distill-qwen")