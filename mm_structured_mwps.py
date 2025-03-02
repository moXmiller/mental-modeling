from mathworld_repo.mathworld.worldmodel.mwp import MWP
from mathworld_repo.mathworld.worldmodel.state import State
from mathworld_repo.mathworld.worldmodel.container import Container
from mathworld_repo.mathworld.worldmodel.tuple import EntityTuple
from mathworld_repo.mathworld.worldmodel.relation import Relation, Transfer, PartWhole, ExplicitAdd

from mm_utils import retrieve_inputs
from mm_text_to_lf import fill_logical_form, load_temp_table, create_re_temp

import re
import copy
import argparse

def create_re_lf(temp):
    re_lfs = []
    coarse_re_lfs = []
    def conditional_replace(match):
        content = match.group()
        if content == "[NUM]": return r"(\d+)"
        else: return r"([\w\s\-]+)"

    pattern = r"\[([^\]]+)\][a-zA-Z]?+"
    for lf in temp["lf"]:
        coarse_lf = lf.replace("None", "[NONE]")
        for f in [lf,coarse_lf]:
            re_lf = re.sub(pattern, conditional_replace, f)
            re_lf = re_lf.lstrip()
            re_lf = re_lf.replace(" ( ", " \( ")
            re_lf = re_lf.replace(" )", " \)")
            if f == lf: re_lfs.append(re_lf)
            else: coarse_re_lfs.append(re_lf)

    temp["re_lf"] = re_lfs
    temp["coarse_re_lf"] = coarse_re_lfs
    return temp

def match_logical_form(filled_lf, temp):
    re_lfs = temp["re_lf"]
    for idx, pattern in enumerate(re_lfs):
        if re.match(pattern, str(filled_lf)): return idx

def process_lf_match(index, temp, lf, span_idx, prev_containers):
    lf_pattern = temp["coarse_re_lf"][index]
    lf_match = re.match(lf_pattern, lf)
    lf_type = temp["type"][index]
    if lf_type == "container":
        label = lf_match.group(1).lower().strip()
        num = int(lf_match.group(2))
        e_tuple = EntityTuple(lf_match.group(3), lf_match.group(4), lf_match.group(5))
        cont = Container(label + "_" + str(span_idx), label, e_tuple.entity, num, e_tuple.attribute, e_tuple.unit, e_tuple)
        return cont, label, num, e_tuple
    if lf_type == "add":
        raise NotImplementedError
    if lf_type == "transfer":
        r_label = lf_match.group(1).lower().strip()
        s_label = lf_match.group(2).lower().strip()
        prev_r = r_label + "_" + str(span_idx-1)
        prev_s = s_label + "_" + str(span_idx-1)
        trans_num = int(lf_match.group(3))
        trans_tuple = EntityTuple(lf_match.group(4), lf_match.group(5), lf_match.group(6))
        if prev_r not in list(prev_containers): trans_label, sign_qu = s_label, - trans_num
        elif prev_s not in list(prev_containers): trans_label, sign_qu = r_label, trans_num
        else: trans_label = None
        if trans_label == None: 
            relation_id = str(s_label) + "_" + str(r_label) + "_" + str(span_idx)
            s_c = prev_containers[prev_s]
            r_c = prev_containers[prev_r]
            new_s = Container(s_label + "_" + str(span_idx), s_label, trans_tuple.entity, None, tuple=trans_tuple)
            new_r = Container(r_label + "_" + str(span_idx), r_label, trans_tuple.entity, None, tuple=trans_tuple)
            new_s.set_value(s_c.quantity.get_value() - trans_num)
            new_r.set_value(r_c.quantity.get_value() + trans_num)
            s_rel = Transfer(relation_id, s_c, new_s, trans_num, trans_tuple, r_label, s_label)
            r_rel = Transfer(relation_id, r_c, new_r, trans_num, trans_tuple, r_label, s_label)
            return s_c, s_rel, new_s, r_c, r_rel, new_r, None, None, None
        else: 
            relation_id = trans_label + "_" + str(span_idx)
            prev_t = trans_label + "_" + str(span_idx-1)
            trans_c = prev_containers[prev_t]
            new_trans = Container(trans_label + "_" + str(span_idx), trans_label, trans_tuple.entity, None, tuple=trans_tuple)
            new_trans.set_value(trans_c.quantity.get_value() + sign_qu)
            if prev_r not in list(prev_containers): 
                trans_rel = Transfer(relation_id, trans_c, new_trans, trans_num, trans_tuple, sender = trans_label)
            elif prev_s not in list(prev_containers): 
                trans_rel = Transfer(relation_id, trans_c, new_trans, trans_num, trans_tuple, recipient = trans_label)
            return None, None, None, None, None, None, trans_c, trans_rel, new_trans

def body_list(body):
    b_list = body.split(". ")
    final_b = b_list[-1]
    b_list = [b + "." for b in b_list if b != b_list[-1]]
    b_list.append(final_b)
    return b_list    

def partial_msp(body, q, index, problem_type, temp, depth):
    if problem_type == "part": part_whole = True
    else: part_whole = False
    b_list = body_list(body)
    mwp_name = problem_type + "_" + str(depth) + "_" + str(index)
    if isinstance(q, list): mwp = MWP(mwp_name, body, q[index][0], b_list)
    elif isinstance(q, str): mwp = MWP(mwp_name, body, q, b_list)
    else: raise ValueError("q is neither list nor string!")
    prev_containers = {}
    for i, s in enumerate(mwp.spans):
        if i == 0: state = State(mwp_name, s)
        else: 
            prev_state = mwp.states[i-1]
            state = copy.deepcopy(prev_state)
            state.span = s
            prev_containers = prev_state.containers
        lf = fill_logical_form(s, part_whole=part_whole)
        match_idx = match_logical_form(lf, temp)
        if temp["type"][match_idx] == "container":
            cont, _, _, _ = process_lf_match(match_idx, temp, lf, i, prev_containers)
            state.add_container(cont)
        if temp["type"][match_idx] == "add":
            raise NotImplementedError
        if temp["type"][match_idx] == "transfer":
            _, s_rel, new_s, _, r_rel, new_r, _, t_rel, new_t = process_lf_match(match_idx, temp, lf, i, prev_containers)
            if t_rel == None:
                state.add_container(new_s)
                state.add_container(new_r)
                state.add_relation(s_rel)
                state.add_relation(r_rel)
            else:
                state.add_container(new_t)
                state.add_relation(t_rel)
        if temp["type"][match_idx] == "times":
            raise NotImplementedError
        mwp.add_state(state)
    return mwp

def create_msp(temp, depth = 6, problem_type = 'trans', sample_size = 400):
    body, questions, answers = retrieve_inputs(depth, problem_type, sample_size, random=False)

    questions = questions.to_list()
    
    mwps = []
    for idx, bdy in enumerate(body):
        mwp = partial_msp(bdy, questions, idx, problem_type, temp, depth)
        mwps.append(mwp)
    return mwps

def sub_msp(temp, body, question, idx, problem_type, depth):
    b_list = body_list(body)
    mwp_name = problem_type + "_" + str(depth) + "_" + str(idx)
    mwp = MWP(mwp_name, body, question, b_list)
    prev_containers = {}
    intermediate_answers = []

    if problem_type == "part": part_whole = True
    else: part_whole = False

    for i, s in enumerate(mwp.spans):
        state = State(mwp_name, s)
        if i > 0: 
            prev_state = mwp.states[i-1]
            state.span = s
            prev_containers = prev_state.containers
        lf = fill_logical_form(s, part_whole=part_whole)
        match_idx = match_logical_form(lf, temp)
        if temp["type"][match_idx] == "container":
            cont, _, _, _ = process_lf_match(match_idx, temp, lf, i, prev_containers)
            state.add_container(cont)
        if temp["type"][match_idx] == "add":
            prev_c, add_cont, add_rel, sign_qu = process_lf_match(match_idx, temp, lf, i, prev_containers)
            state.add_container(prev_c)
            state.add_container(add_cont)
            state.add_relation(add_rel)
            state.update_container(add_cont.id, int(sign_qu))
            intermediate_answers.append(add_cont.quantity.get_value())
        if temp["type"][match_idx] == "transfer":
            prev_s, s_rel, new_s, prev_r, r_rel, new_r, prev_t, t_rel, new_t = process_lf_match(match_idx, temp, lf, i, prev_containers)
            if t_rel == None:
                if prev_s.id not in state.containers.keys():
                    state.add_container(prev_s)
                    state.add_container(prev_r)
                state.add_container(new_s)
                state.add_container(new_r)
                state.add_relation(s_rel)
                state.add_relation(r_rel)
                intermediate_answers.append(new_r.quantity.get_value())
            else:
                state.add_container(prev_t)
                state.add_container(new_t)
                state.add_relation(t_rel)
                intermediate_answers.append(new_t.quantity.get_value())
        mwp.add_state(state)
    return mwp, intermediate_answers

if __name__ == "main":

    parser = argparse.ArgumentParser(description="Slurm job with problem_type and depth arguments")
    parser.add_argument("--problem_type", type=str, choices=["add","trans","part","nonlinear"], required=True, help="problem_type argument")
    parser.add_argument("--depth", type=int, required=True, help="depth argument")
    args = parser.parse_args()

    problem_type = args.problem_type
    depth = args.depth

    if problem_type != "part":
        template_file = 'data/temp_no_plural.csv'
        temp = load_temp_table(template_file)
        temp = create_re_temp(temp)
        temp = create_re_lf(temp)
    else: raise NotImplementedError
    
    print(f"Starting with problem type {problem_type} at depth {depth}.")

    msp = create_msp(temp, depth = depth, problem_type = problem_type, sample_size = 400)