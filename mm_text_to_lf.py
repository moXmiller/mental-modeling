import re
import pandas as pd
import json

def load_temp_table(dir):
    col_names = ["type", "lf", "temp"]
    temp = pd.read_csv(dir, names=col_names)
    return temp

def extract_unique_concepts(temp):
    concepts_set = set()
    for lf in temp["lf"]:
        concepts = re.findall(r"\[([^\]]+)\]", lf)
        for concept in concepts:
            concepts_set.add(concept)
    return list(concepts_set)

def create_re_temp(temp):
    re_temp = []
    def conditional_replace(match):
        content = match.group()
        if content == "[NUM]": return r"(\d+)"
        else: return r"([\w\s\-]+)"

    pattern = r"\[([^\]]+)\][a-zA-Z]?+"
    for tmp in temp["temp"]:
        re_tmp = re.sub(pattern, conditional_replace, tmp)
        re_tmp = re_tmp.lstrip()
        re_tmp = re_tmp
        re_temp.append(re_tmp)
    temp["re_temp"] = re_temp
    return temp

def map_concepts_to_path(concepts, part_whole = False):
    concepts_to_path = {}
    entities = []
    units = []
    unit_to_entity = {}
    pattern = r"([A-Z])_UNIT"
    for concept in concepts:
        if "ENTITY" in concept:
            concepts_to_path[concept] = "data/entities_plural.csv"
            entities.append(f"[{concept}]")
        elif "UNIT" in concept:
            concepts_to_path[concept] = "data/units.csv"
            match = re.match(pattern, concept)
            if match:
                prefix = match.group(1)
                corresponding_entity = f"{prefix}_ENTITY"
                if (corresponding_entity in concepts):
                    unit_to_entity[f"[{concept}]"] = f"[{corresponding_entity}]"
            elif concept == "UNIT":
                unit_to_entity[f"[{concept}]"] = "[ENTITY]"
            units.append(f"[{concept}]")
        elif "ATTRIBUTE" in concept:
            concepts_to_path[concept] = "data/attributes.csv"
        elif concept == "NUM":
            concepts_to_path[concept] = ""
        else:
            concepts_to_path[concept] = "data/more_names.csv"
        entity_to_unit = {v: k for k, v in unit_to_entity.items()}
    return concepts_to_path, entities, units, entity_to_unit

def unit_existence(target_concepts, unit_lfs):
    if any(target_concept in unit_lfs for target_concept in target_concepts): return True
    else: return False

def match_unit_entity(entity_lf, lf, ent_unit_names, entity_to_unit, match, index):
    element = match.group(index+1)
    if element not in ent_unit_names: return lf
    elif element in ent_unit_names:
        unit = ent_unit_names[element]
        unit_lf = entity_to_unit[entity_lf]
        lf = lf.replace(unit_lf, unit + "s")
        lf = lf.replace(entity_lf, element)
        return lf

def check_complete_temp(temp, index):
    tmp = temp["temp"][index]
    lf = temp["lf"][index]
    pattern = r"\[([A-Z0-9_]+)\]"
    tmp_concepts = set(re.findall(pattern,  tmp))
    lf_concepts = set(re.findall(pattern, lf))
    inferred_concepts = []
    if tmp_concepts != lf_concepts: inferred_concepts = list(tmp_concepts ^ lf_concepts)
    return inferred_concepts

def infer_concept(concept, target_concepts):
    pattern = r"\[[A-Z]+_([A-Z]+)\]"
    match = re.match(pattern, f"[{concept}]")
    if match:
        target_label = match.group(1)
        matching_pattern = rf"\[[A-Z]+_{target_label}\]"
        matching_concept = [concept for concept in target_concepts if re.match(matching_pattern, concept)][0]
        return matching_concept
    else: pass

def read_lines_from_path(target_path):
    with open(target_path) as f:
        return [line.strip() for line in f.readlines()]

def load_json(json_path):
    with open(json_path) as json_file:
        return json.load(json_file)

def pars_ad_totum(part_whole_path = "data/part_whole_entities.json"):
    part_whole_entities = load_json(part_whole_path)
    pat_dict = {v: k for k, vs in part_whole_entities.items() for v in vs}
    return pat_dict

def replace_wide_entities(concept, lf, match, index, pat_dictionary):
    element = match.group(index+1)
    if element not in pat_dictionary: return lf
    else:
        pw_entity = pat_dictionary[element]
        lf = lf.replace(concept, pw_entity)
        return lf

def extract_ctc(words):
    combinations_to_check = []
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            comb = " ".join(words[start:end])
            combinations_to_check.append(comb)
    return combinations_to_check

def replace_unit_combination(combinations_to_check, name_dict, translation_dict, concept, lf):
    for combination in combinations_to_check:
        if combination in name_dict:
            unit = name_dict[combination]
            unit_lf = translation_dict[concept]
            lf = lf.replace(unit_lf, unit + "s")
            lf = lf.replace(concept, combination)
    return lf

def replace_part_whole_combination(combinations_to_check, part_whole_dict, concept, lf):
    for combination in combinations_to_check:
        if combination in part_whole_dict:
            pw_entity = part_whole_dict[combination]
            lf = lf.replace(concept, pw_entity)
    return lf                                

def choose_from_valid(combinations_to_check, target_path, lf, concept):
    lines = read_lines_from_path(target_path)                                        
    valid_combinations = []
    for combination in combinations_to_check:                                        
        if combination in lines: valid_combinations.append(combination)                                    
    if valid_combinations != []: lf = lf.replace(f"[{concept}]", max(valid_combinations, key=len))
    return lf

def fill_logical_form(text, ent_unit_path = "data/ent_unit_names.json", part_whole = False):
    if part_whole == True: 
        template_file="data/temp_part_whole.csv"
        part_whole_path = "data/part_whole_entities.json"
        pat_dict = pars_ad_totum(part_whole_path)
    else: template_file = 'data/temp_no_plural.csv'
    temp = load_temp_table(template_file)
    temp = create_re_temp(temp)
    concepts = extract_unique_concepts(temp)
    ent_unit_names = load_json(ent_unit_path)
    concepts_to_path, entity_lfs, unit_lfs, entity_to_unit = map_concepts_to_path(concepts, part_whole)
    candidate_lfs = []
    for idx, pattern in enumerate(temp["re_temp"]):
        # if idx != 22: continue
        mtch = re.match(pattern, str(text))
        if mtch: 
            # print(f"idx match: {idx}")
            # print(temp["temp"][idx])
            inferred_concepts = check_complete_temp(temp, idx)
            concept_pattern = r"\[([A-Z0-9_]+)\]"
            target_concepts = re.findall(concept_pattern, temp["temp"][idx])
            target_paths = [f"{concepts_to_path[concept]}" for concept in target_concepts]
            target_concepts = [f"[{concept}]" for concept in target_concepts]
            
            unit_exists = unit_existence(target_concepts, unit_lfs)

            filled_logical_form = temp["lf"][idx]

            btb_pattern = r"(?:\[(?!NUM)[A-Z0-9_]+\]\s){1,}\[(?!NUM)[A-Z0-9_]+\]"
            btb_list = re.findall(btb_pattern, temp["temp"][idx])
            if btb_list == []:
                for i, concept in enumerate(target_concepts):
                    target_path = target_paths[i]
                    if concept in unit_lfs: pass
                    elif (concept in entity_lfs) and (unit_exists) and (part_whole == False): 
                        filled_logical_form = match_unit_entity(concept, filled_logical_form, ent_unit_names, entity_to_unit, mtch, i)
                    elif (concept in entity_lfs) and (part_whole == True):
                        filled_logical_form = replace_wide_entities(concept, filled_logical_form, mtch, i, pat_dict)
                    elif concept != "[NUM]":
                        lines = read_lines_from_path(target_path)
                        try: element = mtch.group(i+1)
                        except IndexError: continue
                        if element in lines: filled_logical_form = filled_logical_form.replace(concept, mtch.group(i+1))
                    else: 
                        filled_logical_form = filled_logical_form.replace(concept, mtch.group(i+1))
                    try: int(mtch.group(i))
                    except ValueError: pass 
            elif btb_list != []:
                btb_list = [element.split(" ") for element in btb_list]
                btb_set = set(element for btb in btb_list for element in btb)
                for i, concept in enumerate(target_concepts):
                    target_path = target_paths[i]
                    if concept in unit_lfs: pass
                    elif concept != "[NUM]":
                        if concept in btb_set:
                            btb_sub = next((sublist for sublist in btb_list if concept in sublist), None)
                            if concept == btb_sub[0]:
                                elements = []
                                for btb_idx, btb_concept in enumerate(btb_sub):
                                    element = mtch.group(i+btb_idx+1)
                                    elements.append(element)
                                words = [element for element in " ".join(elements).split(" ")]
                                combinations_to_check = extract_ctc(words)
                                for btb_concept in btb_sub:
                                    if (btb_concept in entity_lfs) and (unit_exists) and (part_whole == False):
                                        filled_logical_form = replace_unit_combination(combinations_to_check, ent_unit_names, entity_to_unit, btb_concept, filled_logical_form)
                                    elif (btb_concept in entity_lfs) and (part_whole == True):
                                        filled_logical_form = replace_part_whole_combination(combinations_to_check, pat_dict, btb_concept, filled_logical_form)
                                    elif btb_concept not in unit_lfs:
                                        btb_concept = re.search(r"\[([A-Z_]+)\]", btb_concept).group(1)
                                        target_path = concepts_to_path[btb_concept]
                                        filled_logical_form = choose_from_valid(combinations_to_check, target_path, filled_logical_form, btb_concept)
                        elif (concept in entity_lfs) and unit_exists and not part_whole:
                            filled_logical_form = match_unit_entity(concept, filled_logical_form, ent_unit_names, entity_to_unit, mtch, i)
                        else:    
                            lines = read_lines_from_path(target_path)
                            element = mtch.group(i+1)
                            if element not in lines: continue
                            else: filled_logical_form = filled_logical_form.replace(concept, mtch.group(i+1))
                    else: filled_logical_form = filled_logical_form.replace(concept, mtch.group(i+1))
                    try: int(mtch.group(i))
                    except ValueError: pass
            remaining_concepts = re.findall(concept_pattern, filled_logical_form)

            diff_rem_inf = set(remaining_concepts) ^ set(inferred_concepts)
            for ri in diff_rem_inf:
                if "UNIT" in ri: inferred_concepts.append(ri)
            if (set(remaining_concepts) ^ set(inferred_concepts)) != set(): filled_logical_form = ""
            elif remaining_concepts != []: 
                for rc in remaining_concepts:                    
                    if len(remaining_concepts) > 1:
                        elements = []
                        inferred_concepts = [infer_concept(rc, target_concepts) for rc in remaining_concepts]
                        for j, tc in enumerate(target_concepts, start=1):
                            if tc in inferred_concepts:
                                element = mtch.group(j)
                                elements.append(element)
                        words = [element for element in " ".join(elements).split(" ")]
                        combinations_to_check = extract_ctc(words)
                        if ((rc in entity_lfs) or (f"[{rc}]" in entity_lfs)) and unit_exists and not part_whole:
                            if rc not in entity_lfs: rc = f"[{rc}]"
                            filled_logical_form = replace_unit_combination(combinations_to_check, ent_unit_names, entity_to_unit, rc, filled_logical_form)
                        elif ((rc in entity_lfs) or (f"[{rc}]" in entity_lfs)) and part_whole:
                            if rc not in entity_lfs: rc = f"[{rc}]"
                            filled_logical_form = replace_part_whole_combination(combinations_to_check, pat_dict, rc, filled_logical_form)
                        elif rc not in unit_lfs:
                            target_path = concepts_to_path[rc]
                            filled_logical_form = choose_from_valid(combinations_to_check, target_path, filled_logical_form, rc)                
                    else:
                        inferred_concept = infer_concept(rc, target_concepts)
                        inferred_element = next((mtch.group(j) for j, tc in enumerate(target_concepts, start=1) if tc == inferred_concept), None)
                        filled_logical_form = filled_logical_form.replace(f"[{rc}]", inferred_element)
            if filled_logical_form != "": 
                candidate_lfs.append(filled_logical_form)
    final_lf = max(candidate_lfs, key=lambda x: x.count(","))
    return final_lf