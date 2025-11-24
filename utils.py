import os
# Avoid transformers importing torchvision (which can fail due to mismatched builds)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import random
import json
from typing import Optional, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from tqdm import tqdm, trange
from collections import defaultdict

from config import prompt_formats, sycophancy_phrases, MODEL_CONFIGS, DATASET_CONFIGS

def load_model(model_path: str, device: str = "cuda") -> PreTrainedModel:
    if model_path in MODEL_CONFIGS:
        model_path = MODEL_CONFIGS[model_path]["path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    ).eval()
    
    print(f"Hidden size: {model.config.hidden_size} | layers: {model.config.num_hidden_layers}")

    return model, tokenizer

def set_seed(seed: int) -> None:
    """Minimal, reliable seeding for Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return

def load_json_data(file_path: str, max_samples: Optional[int] = None) -> list:
    print(f"Loading data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    print(f"Loaded {len(data)} samples")
    return data

def sample_and_order(spectrum: List[str]) -> Tuple[str, str, int, int]:
    """Sample two distinct phrases from ordered spectrum and return (low_text, high_text, low_idx, high_idx)."""
    i, j = random.sample(range(len(spectrum)), k=2)
    lo, hi = (i, j) if i < j else (j, i)
    return spectrum[lo], spectrum[hi]

def sample_syc_phrase(pool: list[str]) -> str:
    return random.choice(pool)

def get_political_leaning_labels(partisan_lean: str):
    if partisan_lean == "Republican":
        return 1
    elif partisan_lean == "Democratic":
        return 0
    else:
        return -1

def get_sexual_explicit_bias_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []

    for item in data:

        truth_label = item["sexual_explicit"]
        bias_label = item["toxicity"]

        formatted_prompt = prompt_formats["POINTWISE_SEXUAL_EXPLICIT_BIAS_FORMAT"]
        formatted_prompt = formatted_prompt.format(text=item["text"])

        positive_prompt = formatted_prompt + " True"
        negative_prompt = formatted_prompt + " False"

        if random.random() < 0.5:
            prompts_group_1.append(positive_prompt)
            prompts_group_2.append(negative_prompt)
            truth_labels.append(truth_label)
            bias_labels.append(bias_label)
        else:
            prompts_group_1.append(negative_prompt)
            prompts_group_2.append(positive_prompt)
            truth_labels.append(not truth_label)
            bias_labels.append(not bias_label)

    return prompts_group_1, prompts_group_2, truth_labels, bias_labels
    
def get_toxicity_bias_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []

    for item in data:

        truth_label = item["toxicity"]
        bias_label = item["sexual_explicit"]

        formatted_prompt = prompt_formats["POINTWISE_TOXICITY_BIAS_FORMAT"]
        formatted_prompt = formatted_prompt.format(text=item["text"])

        positive_prompt = formatted_prompt + " True"
        negative_prompt = formatted_prompt + " False"

        if random.random() < 0.5:
            prompts_group_1.append(positive_prompt)
            prompts_group_2.append(negative_prompt)
            truth_labels.append(truth_label)
            bias_labels.append(bias_label)
        else:
            prompts_group_1.append(negative_prompt)
            prompts_group_2.append(positive_prompt)
            truth_labels.append(not truth_label)
            bias_labels.append(not bias_label)

    return prompts_group_1, prompts_group_2, truth_labels, bias_labels

def get_political_leaning_labels(partisan_lean: str):
    if partisan_lean == "Republican":
        return 1
    elif partisan_lean == "Democratic":
        return 0
    else:
        return -1

def get_political_leaning_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []
    for item in data:
        political_leaning_label = get_political_leaning_labels(item["partisan_lean"])

        if political_leaning_label != -1:
            formatted_prompt = prompt_formats["POINTWISE_POLITICAL_LEANING_FORMAT"]
            formatted_prompt = formatted_prompt.format(text=item["text"])
            truth_label = item["labels"]

            positive_prompt = formatted_prompt + " True"
            negative_prompt = formatted_prompt + " False"

            if random.random() < 0.5:
                prompts_group_1.append(positive_prompt)
                prompts_group_2.append(negative_prompt)
                truth_labels.append(truth_label)
                bias_labels.append(political_leaning_label)
            else:
                prompts_group_1.append(negative_prompt)
                prompts_group_2.append(positive_prompt)
                truth_labels.append(not truth_label)
                bias_labels.append(not political_leaning_label)

    return prompts_group_1, prompts_group_2, truth_labels, bias_labels

def get_political_leaning_log_probs_prompts(data: list) -> list:
    prompts = []
    truth_labels = []
    bias_labels = []
    for item in data:
        political_leaning_label = get_political_leaning_labels(item["partisan_lean"])

        if political_leaning_label != -1:
            formatted_prompt = prompt_formats["POINTWISE_POLITICAL_LEANING_FORMAT"]
            formatted_prompt = formatted_prompt.format(text=item["text"])
            truth_label = item["labels"]

            prompts.append(formatted_prompt)
            truth_labels.append(truth_label)
            bias_labels.append(political_leaning_label)
            

    return prompts, truth_labels, bias_labels

def get_punctuation_bias_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []
    for item in data:

        truth_label = item["label"]
        formatted_prompt = prompt_formats["POINTWISE_PUNCTUATION_BIAS_FORMAT"]

        if random.random() < 0.5:
            item_choice = add_lots_of_punctuation(item["choice"])
            bias_label = 1
        else:
            item_choice = item["choice"]
            bias_label = 0

        
        formatted_prompt = formatted_prompt.format(question=item["question"], choice=item_choice)
        

        positive_prompt = formatted_prompt + " True"
        negative_prompt = formatted_prompt + " False"

        prompts_group_1.append(positive_prompt)
        prompts_group_2.append(negative_prompt)
        truth_labels.append(truth_label)
        bias_labels.append(bias_label)


    return prompts_group_1, prompts_group_2, truth_labels, bias_labels

def get_pairwise_punctuation_bias_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []
    for item in data:

        truth_label = item["label"]
        formatted_prompt = prompt_formats["POINTWISE_PUNCTUATION_BIAS_FORMAT"]

        if random.random() < 0.5:
            item_choice = add_lots_of_punctuation(item["choice"])
            item_choice_2 = item["choice_2"]
            bias_label = 1
        else:
            item_choice = item["choice"]
            item_choice_2 = add_lots_of_punctuation(item["choice_2"])
            bias_label = 0

        
        positive_prompt = formatted_prompt.format(question=item["question"], choice=item_choice)
        negative_prompt = formatted_prompt.format(question=item["question"], choice=item_choice_2)

        prompts_group_1.append(positive_prompt)
        prompts_group_2.append(negative_prompt)
        truth_labels.append(truth_label)
        bias_labels.append(bias_label)


    return prompts_group_1, prompts_group_2, truth_labels, bias_labels

def get_tense_bias_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []
    for item in data:

        truth_label = item["label"]
        formatted_prompt = prompt_formats["POINTWISE_PUNCTUATION_BIAS_FORMAT"]

        if random.random() < 0.5:
            item_choice = to_past_tense(item["choice"])
            bias_label = 1
        else:
            item_choice = to_future_tense(item["choice"])
            bias_label = 0

        
        formatted_prompt = formatted_prompt.format(question=item["question"], choice=item_choice)
        

        positive_prompt = formatted_prompt + " True"
        negative_prompt = formatted_prompt + " False"

        prompts_group_1.append(positive_prompt)
        prompts_group_2.append(negative_prompt)
        truth_labels.append(truth_label)
        bias_labels.append(bias_label)


    return prompts_group_1, prompts_group_2, truth_labels, bias_labels

def get_pairwise_tense_bias_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []
    for item in data:

        truth_label = item["label"]
        formatted_prompt = prompt_formats["POINTWISE_PUNCTUATION_BIAS_FORMAT"]

        if random.random() < 0.5:
            item_choice = to_past_tense(item["choice"])
            item_choice_2 = to_future_tense(item["choice_2"])
            bias_label = 1
        else:
            item_choice = to_future_tense(item["choice"])
            item_choice_2 = to_past_tense(item["choice_2"])
            bias_label = 0

        
        positive_prompt = formatted_prompt.format(question=item["question"], choice=item_choice)
        negative_prompt = formatted_prompt.format(question=item["question"], choice=item_choice_2)

        prompts_group_1.append(positive_prompt)
        prompts_group_2.append(negative_prompt)
        truth_labels.append(truth_label)
        bias_labels.append(bias_label)


    return prompts_group_1, prompts_group_2, truth_labels, bias_labels

def get_sycophancy_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []

    groups = defaultdict(list)
    for item in tqdm(data):
        groups[item["question"]].append(item)

    positive_pool = sycophancy_phrases["POS_SYC_POOL"]
    negative_pool = sycophancy_phrases["NEG_SYC_POOL"]

    for group_id, group in tqdm(groups.items()):
        item = random.choice(group)

        negative_sycophancy = sample_syc_phrase(negative_pool)
        positive_sycophancy = sample_syc_phrase(positive_pool)

        prompt_template = prompt_formats["POINTWISE_SYCOPHANCY_FORMAT"]

        if random.random() < 0.5:
            bias_label = 1

            truth_label = item["label"]
            question = item["question"]
            answer = item["answer"] if "answer" in item else item["choice"]
            choice = f"{positive_sycophancy} {item['choice']}"

            formatted_prompt = prompt_template.format(question=question, answer=answer, choice=choice)
            positive_prompt = formatted_prompt + " True"
            negative_prompt = formatted_prompt + " False"

            if random.random() < 0.5:
                prompts_group_1.append(positive_prompt)
                prompts_group_2.append(negative_prompt)
                truth_labels.append(truth_label)
                
            else:
                prompts_group_1.append(negative_prompt)
                prompts_group_2.append(positive_prompt)
                truth_labels.append(1 - truth_label)

            bias_labels.append(bias_label)
        else:
            bias_label = 0

            truth_label = item["label"]
            question = item["question"]
            answer = item["answer_2"] if "answer_2" in item else item["choice_2"]
            choice = f"{negative_sycophancy} {item['choice']}"

            formatted_prompt = prompt_template.format(question=question, answer=answer, choice=choice)
            positive_prompt = formatted_prompt + " True"
            negative_prompt = formatted_prompt + " False"

            if random.random() < 0.5:
                prompts_group_1.append(positive_prompt)
                prompts_group_2.append(negative_prompt)
                truth_labels.append(truth_label)
                
            else:
                prompts_group_1.append(negative_prompt)
                prompts_group_2.append(positive_prompt)
                truth_labels.append(1 - truth_label)

            bias_labels.append(bias_label)

    return prompts_group_1, prompts_group_2, truth_labels, bias_labels


def get_prompts_and_answers(data: list) -> list:
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []
    bias_labels = []

    groups = defaultdict(list)
    for item in tqdm(data):
        groups[item["question"]].append(item)

    for group_id, group in tqdm(groups.items()):
        item = random.choice(group)
        prompt_template = prompt_formats["POINTWISE_FORMAT"]
        formatted_prompt = prompt_template.format(question=item["question"], choice=item["choice"])
        positive_prompt = formatted_prompt + " True"
        negative_prompt = formatted_prompt + " False"   
        prompts_group_1.append(positive_prompt)
        prompts_group_2.append(negative_prompt)
        truth_labels.append(item["label"])
        # bias_labels.append(item["bias_label"])

    return prompts_group_1, prompts_group_2, truth_labels, bias_labels
            

def get_pairwise_prompts_and_answers(data: list, prompt_format: str, sycophancy_type: str = "pooling") -> list:
    formatted_prompt = prompt_formats[prompt_format]
    prompts_group_1 = []
    prompts_group_2 = []
    truth_labels = []

    from collections import defaultdict
    groups = defaultdict(list)
    # print(type(data))

    # print(data[0])
    # print(data[1])
    
    for item in tqdm(data):
        groups[item["question"]].append(item)

    # print(len(groups))

    if "SYCOPHANCY" in prompt_format:
        if sycophancy_type == "pooling":
            positive_pool = sycophancy_phrases["POS_SYC_POOL"]
            negative_pool = sycophancy_phrases["NEG_SYC_POOL"]
        elif sycophancy_type == "spectrum":
            spectrum = sycophancy_phrases["SYCOPHANCY_SPECTRUM"]
            

    GROUP_A, GROUP_B, y_truth, y_bias_label = [], [], [], []
    for group_id, group in tqdm(groups.items()):
        # assert len(group) == 2 and (group[0]["label"] != group[1]["label"]), f"bad group {group_id}"
        if "SYCOPHANCY" not in prompt_format and "COMPLETION_BIAS" not in prompt_format and "PUNCTUATION_BIAS" not in prompt_format and "TENSE_BIAS" not in prompt_format:
            A = {
                "question": group[0]["question"],
                "choice": group[0]["choice"],
                "choice_2": group[0]["choice_2"],
                "label": group[0]["label"]
            }
            B = {
                "question": group[0]["question"], 
                "choice": group[0]["choice_2"],
                "choice_2": group[0]["choice"],
                "label": not group[0]["label"]
            }
            
            if random.random() < 0.5:
                GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"])
            else:
                GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"])

        elif "TENSE_BIAS" in prompt_format:
            print("Doing TENSE BIAS")
            A = {
                "question": group[0]["question"],
                "choice": group[0]["choice"],
                "choice_2": group[0]["choice_2"],
                "label": group[0]["label"]
            }
            B = {
                "question": group[0]["question"], 
                "choice": group[0]["choice_2"],
                "choice_2": group[0]["choice"],
                "label": not group[0]["label"]
            }
            
            if random.random() < 0.5:
                # if random.random() < 0.5:
                #     GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(1)
                # else:
                #     GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(0)

                if random.random() < 0.5:
                    GROUP_A.append(to_past_tense(formatted_prompt.format(**A))), GROUP_B.append(to_future_tense(formatted_prompt.format(**B))), y_truth.append(A["label"]), y_bias_label.append(1)
                else:
                    GROUP_A.append(to_future_tense(formatted_prompt.format(**A))), GROUP_B.append(to_past_tense(formatted_prompt.format(**B))), y_truth.append(A["label"]), y_bias_label.append(0)
            else:
                # if random.random() < 0.5:
                #     GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)
                # else:
                #     GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(0)

                if random.random() < 0.5:
                    GROUP_A.append(to_past_tense(formatted_prompt.format(**B))), GROUP_B.append(to_future_tense(formatted_prompt.format(**A))), y_truth.append(B["label"]), y_bias_label.append(1)
                else:
                    GROUP_A.append(to_future_tense(formatted_prompt.format(**B))), GROUP_B.append(to_past_tense(formatted_prompt.format(**A))), y_truth.append(B["label"]), y_bias_label.append(0)

        elif "PUNCTUATION_BIAS" in prompt_format:
            print("Doing PUNCTUATION_BIAS")
            A = {
                "question": group[0]["question"],
                "choice": group[0]["choice"],
                "choice_2": group[0]["choice_2"],
                "label": group[0]["label"]
            }
            B = {
                "question": group[0]["question"], 
                "choice": group[0]["choice_2"],
                "choice_2": group[0]["choice"],
                "label": not group[0]["label"]
            }
            
            if random.random() < 0.5:
                # if random.random() < 0.5:
                #     GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(1)
                # else:
                #     GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(0)

                if random.random() < 0.5:
                    GROUP_A.append(add_lots_of_punctuation(formatted_prompt.format(**A))), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(1)
                else:
                    GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(add_lots_of_punctuation(formatted_prompt.format(**B))), y_truth.append(A["label"]), y_bias_label.append(0)
            else:
                # if random.random() < 0.5:
                #     GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)
                # else:
                #     GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(0)

                if random.random() < 0.5:
                    GROUP_A.append(add_lots_of_punctuation(formatted_prompt.format(**B))), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)
                else:
                    GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(add_lots_of_punctuation(formatted_prompt.format(**A))), y_truth.append(B["label"]), y_bias_label.append(0)
                

            # ## FIRST PAIR OF COMPLETION BIAS
            # response_1 = f"{group[0]['choice']}"
            # response_2 = f"{group[0]['choice_2']}"

            # A = {
            #     "question": group[0]["question"],
            #     "response_1": response_1,
            #     "response_2": response_2,
            #     "label": group[0]["label"]
            # }

            # B = {
            #     "question": group[0]["question"],
            #     "response_1": response_2,
            #     "response_2": response_1,
            #     "label": not group[0]["label"]
            # }
            
            # if random.random() < 0.5:
            #     GROUP_A.append(add_lots_of_punctuation(formatted_prompt.format(**A))), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(1)
            # else:
            #     GROUP_A.append(add_lots_of_punctuation(formatted_prompt.format(**B))), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)

            # ## SECOND PAIR OF COMPLETION BIAS
            # response_1 = f"{group[0]['choice_2']}"
            # response_2 = f"{group[0]['choice']}"

            # A = {
            #     "question": group[0]["question"],
            #     "response_1": response_1,
            #     "response_2": response_2,
            #     "label": not group[0]["label"]
            # }
            
            # B = {
            #     "question": group[0]["question"],
            #     "response_1": response_2,
            #     "response_2": response_1,
            #     "label": group[0]["label"]
            # }
            
            # if random.random() < 0.5:
            #     GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(add_lots_of_punctuation(formatted_prompt.format(**B))), y_truth.append(A["label"]), y_bias_label.append(0)
            # else:
            #     GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(add_lots_of_punctuation(formatted_prompt.format(**A))), y_truth.append(B["label"]), y_bias_label.append(0)

        elif "SYCOPHANCY" in prompt_format:
            if sycophancy_type == "pooling":
                negative_sycophancy = sample_syc_phrase(negative_pool)
                positive_sycophancy = sample_syc_phrase(positive_pool)
            elif sycophancy_type == "spectrum":
                negative_sycophancy, positive_sycophancy = sample_and_order(spectrum)

            # FIRST PAIR OF SYCOPHANCY
            answer = group[0]["answer"] if "answer" in group[0] else group[0]["choice"]
            sycophancy_combined_choice = f"{positive_sycophancy} {group[0]['choice']}"
            sycophancy_combined_choice_2 = f"{negative_sycophancy} {group[0]['choice_2']}"

            A = {
                "question": group[0]["question"],
                "answer": answer,
                "sycophancy_combined_choice": sycophancy_combined_choice,
                "sycophancy_combined_choice_2": sycophancy_combined_choice_2,
                "label": group[0]["label"]
            }

            B = {
                "question": group[0]["question"],
                "answer": answer,
                "sycophancy_combined_choice": sycophancy_combined_choice_2,
                "sycophancy_combined_choice_2": sycophancy_combined_choice,
                "label": not group[0]["label"]
            }

            if random.random() < 0.5:
                GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(1)
            else:
                GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(0)

            # SECOND PAIR OF SYCOPHANCY -----------------------------------------------------------
            answer = group[0]["answer_2"] if "answer_2" in group[0] else group[0]["choice_2"]
            sycophancy_combined_choice = f"{negative_sycophancy} {group[0]['choice']}"
            sycophancy_combined_choice_2 = f"{positive_sycophancy} {group[0]['choice_2']}"

            A = {
                "question": group[0]["question"],
                "answer": answer,
                "sycophancy_combined_choice": sycophancy_combined_choice,
                "sycophancy_combined_choice_2": sycophancy_combined_choice_2,
                "label": group[0]["label"]
            }

            B = {
                "question": group[0]["question"],
                "answer": answer,
                "sycophancy_combined_choice": sycophancy_combined_choice_2,
                "sycophancy_combined_choice_2": sycophancy_combined_choice,
                "label": not group[0]["label"]
            }

            if random.random() < 0.5:
                GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(0)
            else:
                GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)
        
        elif "COMPLETION_BIAS" in prompt_format:

            ## FIRST PAIR OF COMPLETION BIAS
            response_1 = f"The answer to the question is {group[0]['answer']}."
            response_2 = f"{group[0]['choice_2']}"

            A = {
                "question": group[0]["question"],
                "response_1": response_1,
                "response_2": response_2,
                "label": group[0]["label"]
            }

            B = {
                "question": group[0]["question"],
                "response_1": response_2,
                "response_2": response_1,
                "label": not group[0]["label"]
            }
            
            if random.random() < 0.5:
                GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(0)
            else:
                GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)

            ## SECOND PAIR OF COMPLETION BIAS
            response_1 = f"The answer to the question is {group[0]['answer_2']}."
            response_2 = f"{group[0]['choice']}"

            A = {
                "question": group[0]["question"],
                "response_1": response_1,
                "response_2": response_2,
                "label": not group[0]["label"]
            }
            
            B = {
                "question": group[0]["question"],
                "response_1": response_2,
                "response_2": response_1,
                "label": group[0]["label"]
            }
            
            if random.random() < 0.5:
                GROUP_A.append(formatted_prompt.format(**A)), GROUP_B.append(formatted_prompt.format(**B)), y_truth.append(A["label"]), y_bias_label.append(0)
            else:
                GROUP_A.append(formatted_prompt.format(**B)), GROUP_B.append(formatted_prompt.format(**A)), y_truth.append(B["label"]), y_bias_label.append(1)

    return GROUP_A, GROUP_B, y_truth, y_bias_label

def prepare_consistency_pairs(positive_prompts: list[str], negative_prompts: list[str], y_truth: list[int], y_sycophancy: list[int], append_boolean_token: bool = True) -> list[str]:
    if append_boolean_token:
        positive_prompts_true = [f"{prompt} True" for prompt in positive_prompts]
        positive_prompts_false = [f"{prompt} False" for prompt in positive_prompts]

        negative_prompts_true = [f"{prompt} True" for prompt in negative_prompts]
        negative_prompts_false = [f"{prompt} False" for prompt in negative_prompts]

        final_positive_prompts = positive_prompts_true + negative_prompts_false
        final_negative_prompts = positive_prompts_false + negative_prompts_true
        final_y = y_truth + y_truth

        return final_positive_prompts, final_negative_prompts, final_y

    return positive_prompts, negative_prompts, y_truth, y_sycophancy

def get_last_token_activations_and_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: torch.device | None = None,
    batch_size: int = 8,
    layers: list[int] | None = None,
) -> tuple[dict[int, torch.Tensor], list[dict[str, float]]]:
    """Returns a dict mapping layer -> activations tensor (n_samples, n_dim) and a list of logprobs dicts (one per prompt)."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    n_samples = len(prompts)
    acts_by_layer = {l: [] for l in layers}
    logprobs_list = []
    for i in trange(0, n_samples, batch_size, desc="Getting activations"):
        batch_prompts = prompts[i:i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        attention_mask_tensor = torch.tensor(encoded["attention_mask"]) if not torch.is_tensor(encoded["attention_mask"]) else encoded["attention_mask"]
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        last_token_indices = torch.sum(attention_mask_tensor, dim=1) - 1
        logprobs = torch.log_softmax(logits, dim=-1)
        for j in range(len(batch_prompts)):
            for l in layers:
                layer_hidden = hidden_states[l]
                act = layer_hidden[j, last_token_indices[j], :].cpu()
                acts_by_layer[l].append(act)
            last_token_logprobs = logprobs[j, last_token_indices[j], :]
            topk = min(20, last_token_logprobs.shape[0])
            top_logprobs_vals, top_logprobs_idx = torch.topk(last_token_logprobs, k=topk)
            logprobs_for_prompt = {}
            for idx, val in zip(top_logprobs_idx.tolist(), top_logprobs_vals.tolist()):
                token_str = tokenizer.decode([idx])
                logprobs_for_prompt[token_str] = val
            logprobs_list.append(logprobs_for_prompt)
        del outputs, hidden_states, logits, logprobs
        torch.cuda.empty_cache()
    for l in layers:
        acts_by_layer[l] = torch.stack(acts_by_layer[l], dim=0)
    return acts_by_layer, logprobs_list

def get_last_token_activations_and_logprobs_reload(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    prompts: list[str] | tuple[list[str], list[str]] | None = None,
    device: torch.device | None = None,
    batch_size: int = 8,
    layers: list[int] | None = None,
    exp_config: dict | None = None,
    save_dir_root: str = "/workspace/activations-final",
    y_truth: list[int] | None = None,
    y_bias: list[int] | None = None,

) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[int], list[int]]:
    """Thin wrapper. Prefer the explicit helpers below.

    - Load-only: provide exp_config only → calls load_last_token_activations
    """
    if prompts is None and exp_config is not None:
        return load_last_token_activations(
            exp_config=exp_config,
            layers=layers,
            save_dir_root=save_dir_root,
        )
    raise RuntimeError("Invalid call: use compute_and_save_last_token_activations(exp_config) to compute, or load_last_token_activations(exp_config) to load.")

def compute_and_save_last_token_activations(
    exp_config: dict,
    device: torch.device | None = None,
    batch_size: int = 8,
    layers: list[int] | None = None,
    save_dir_root: str = "/workspace/activations-final",
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[int], list[int]]:
    """Compute activations from only exp_config: loads model, data, and builds prompts.

    Currently supports sycophancy prompts.
    """
    # Load model/tokenizer from MODEL_CONFIGS
    model_name_cfg = exp_config.get("model_name")
    if not model_name_cfg or model_name_cfg not in MODEL_CONFIGS:
        raise RuntimeError("Invalid or missing model_name in exp_config for compute.")
    model, tokenizer = load_model(model_name_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        device = next(model.parameters()).device
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    # Load data from DATASET_CONFIGS
    dataset_name_cfg = exp_config.get("dataset_name")
    split = exp_config.get("split", "train")
    if not dataset_name_cfg or dataset_name_cfg not in DATASET_CONFIGS:
        raise RuntimeError("Invalid or missing dataset_name in exp_config for compute.")
    split_path = DATASET_CONFIGS[dataset_name_cfg].get(split)
    if not split_path:
        raise RuntimeError(f"No path configured for dataset '{dataset_name_cfg}' split '{split}'.")
    num_samples = exp_config.get("num_train_samples") or exp_config.get("num_samples")
    data = load_json_data(split_path, max_samples=num_samples)

    # Build prompts for sycophancy only (for now)
    prompt_format = exp_config.get("prompt_format", "")
    if prompt_format == "direct_pooling_sycophancy":
        positive_prompts, negative_prompts, y_truth, y_bias = get_sycophancy_prompts_and_answers(data)
    elif prompt_format == "direct":
        positive_prompts, negative_prompts, y_truth, y_bias = get_prompts_and_answers(data)
    else:
        raise RuntimeError("Only sycophancy prompt format is supported in compute for now.")

    def _process_one(prompt_list: list[str]) -> dict[int, torch.Tensor]:
        n_samples_local = len(prompt_list)
        acts_by_layer_local = {l: [] for l in layers}  # type: ignore[arg-type]
        for i in trange(0, n_samples_local, batch_size, desc="Getting activations"):
            batch_prompts = prompt_list[i:i + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            attention_mask_tensor = torch.tensor(encoded["attention_mask"]) if not torch.is_tensor(encoded["attention_mask"]) else encoded["attention_mask"]
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_token_indices = torch.sum(attention_mask_tensor, dim=1) - 1
            for j in range(len(batch_prompts)):
                for l in layers:  # type: ignore[assignment]
                    layer_hidden = hidden_states[l]
                    act = layer_hidden[j, last_token_indices[j], :].cpu()
                    acts_by_layer_local[l].append(act)
            del outputs, hidden_states
            torch.cuda.empty_cache()
        for l in layers:  # type: ignore[assignment]
            acts_by_layer_local[l] = torch.stack(acts_by_layer_local[l], dim=0)
        return acts_by_layer_local

    model_name = exp_config.get("model_name", "unknown-model")
    dataset_name = exp_config.get("dataset_name", "unknown-dataset")
    split_for_path = exp_config.get("split", "unspecified-split")
    prompt_format_for_path = exp_config.get("prompt_format", "unknown-format")
    base_dir = os.path.join(save_dir_root, model_name, dataset_name, split_for_path, prompt_format_for_path)
    os.makedirs(base_dir, exist_ok=True)

    acts_pos = _process_one(positive_prompts)
    acts_neg = _process_one(negative_prompts)

    for split_label, acts in ("positive", acts_pos), ("negative", acts_neg):
        out_dir = os.path.join(base_dir, split_label)
        os.makedirs(out_dir, exist_ok=True)
        for l, tensor in acts.items():
            torch.save(tensor, os.path.join(out_dir, f"layer_{l}.pt"))

    labels_path = os.path.join(base_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({
            "y_truth": list(map(int, y_truth)),
            "y_bias": list(map(int, y_bias)),
            "num_pairs_expected": len(positive_prompts),
            "num_pairs_expected_neg": len(negative_prompts),
        }, f)

    return acts_pos, acts_neg, list(map(int, y_truth)), list(map(int, y_bias))

def load_last_token_activations(
    exp_config: dict,
    layers: list[int] | None = None,
    save_dir_root: str = "/workspace/activations-final",
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[int], list[int]]:
    model_name = exp_config.get("model_name", "unknown-model")
    dataset_name = exp_config.get("dataset_name", "unknown-dataset")
    split = exp_config.get("split", "unspecified-split")
    prompt_format = exp_config.get("prompt_format", "unknown-format")
    base_dir = os.path.join(save_dir_root, model_name, dataset_name, split, prompt_format)

    labels_path = os.path.join(base_dir, "labels.json")
    if not os.path.exists(labels_path):
        raise RuntimeError("Activations and labels for the provided config are not saved; please run compute_and_save_last_token_activations first.")
    with open(labels_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    y_t = meta.get("y_truth", [])
    y_b = meta.get("y_bias", [])

    def _load_split(split_label: str) -> dict[int, torch.Tensor]:
        split_dir = os.path.join(base_dir, split_label)
        if not os.path.isdir(split_dir):
            raise RuntimeError("Activations and labels for the provided config are not saved; please run compute_and_save_last_token_activations first.")
        out: dict[int, torch.Tensor] = {}
        if layers is None:
            for fname in sorted(os.listdir(split_dir)):
                if fname.startswith("layer_") and fname.endswith(".pt"):
                    try:
                        l = int(fname[len("layer_"):-3])
                    except Exception:
                        continue
                    out[l] = torch.load(os.path.join(split_dir, fname), map_location="cpu")
        else:
            for l in layers:
                fpath = os.path.join(split_dir, f"layer_{l}.pt")
                if not os.path.exists(fpath):
                    raise RuntimeError("Activations and labels for the provided config are not saved; please run compute_and_save_last_token_activations first.")
                out[l] = torch.load(fpath, map_location="cpu")
        if not out:
            raise RuntimeError("Activations and labels for the provided config are not saved; please run compute_and_save_last_token_activations first.")
        return out

    acts_pos = _load_split("positive")
    acts_neg = _load_split("negative")
    return acts_pos, acts_neg, y_t, y_b

def compute_mean_and_std(X: torch.Tensor):
    X = X.to(torch.float32)
    mu = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True)
    return mu, std

def normalize_with(mu, std, X):
    X = X.to(torch.float32)
    X_n = (X - mu) / (std + 1e-12)
    return X_n

def compute_mean_and_std_pair(X_pos: torch.Tensor, X_neg: torch.Tensor):
    X_pos = X_pos.to(torch.float32)
    X_neg = X_neg.to(torch.float32)

    mu_pos = X_pos.mean(0, keepdim=True)
    std_pos = X_pos.std(0, keepdim=True)

    mu_neg = X_neg.mean(0, keepdim=True)
    std_neg = X_neg.std(0, keepdim=True)

    return mu_pos, std_pos, mu_neg, std_neg

def normalize_with_pair(mu_pos, std_pos, mu_neg, std_neg, X_pos, X_neg):
    X_pos = X_pos.to(torch.float32)
    X_neg = X_neg.to(torch.float32)

    X_pos_n = (X_pos - mu_pos) / (std_pos + 1e-12)
    X_neg_n = (X_neg - mu_neg) / (std_neg + 1e-12)
    
    return X_pos_n, X_neg_n

import re
import random

def add_lots_of_punctuation(text):
    """
    Takes text and adds lots of punctuation only at the end of each sentence.
    
    Args:
        text (str): The input text to modify
        
    Returns:
        str: The text with enhanced ending punctuation for each sentence
    """
    
    if not text or len(text.strip()) < 3:
        return text
    
    # Split into sentences while preserving the original punctuation
    sentences = re.split(r'([.!?]+)', text)
    
    result = []
    
    # Process sentence pairs (text + punctuation)
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence_text = sentences[i].strip()
            
            if sentence_text:  # Only process non-empty sentences
                # Get original punctuation if it exists
                original_punct = sentences[i + 1] if i + 1 < len(sentences) else '.'
                
                # Add the sentence text
                result.append(sentence_text)
                
                # Add enhanced punctuation based on original type
                enhanced_punct = get_enhanced_punctuation(original_punct)
                result.append(enhanced_punct)
                
                # Add space after sentence (except for last one)
                if i + 2 < len(sentences):
                    result.append(' ')
    
    return ''.join(result)


def get_enhanced_punctuation(original_punct):
    """Generate enhanced punctuation based on the original ending"""
    
    # Determine the type of original punctuation
    if '!' in original_punct:
        # Exclamation sentences get more exciting endings
        endings = ['!', '!!', '!!!', '!?', '...!', '!!?', '!!!']
        return random.choice(endings)
    
    elif '?' in original_punct:
        # Question sentences get enhanced question endings
        endings = ['?', '??', '?!', '...?', '??!', '?!!']
        return random.choice(endings)
    
    else:
        # Regular sentences (periods) get various dramatic endings
        endings = ['.', '!', '!!', '...', '..!', '...!', '.!', '!!']
        return random.choice(endings)

import re, random
from typing import List, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")

STOP = {
    "the","a","an","this","that","these","those","is","are","was","were","be","been","being",
    "to","of","in","on","for","with","by","as","at","it","its","and","or","but","if","then",
    "than","so","because","not","have","has","had","do","does","did","will","would","can",
    "could","should","must","may","might","you","we","they","he","she","i","them","him","her",
    "our","your","their","all","any","only","most","more","less","no","nor","either","neither"
}

PREP = {
    "for","in","on","with","without","under","over","after","before","by","to","from","about",
    "within","during","among","between","against","toward","via","like","as","at","around","near",
    "including","except","into","onto","through","across","beyond","behind","above","below","per"
}

OPERATORS = {"not","never","only","always","must","should","no","none","all","any","cannot","can’t","won’t"}
COORDS = {"and","or","but","nor"}

QUOTE_STYLES = [("“","”"), ("‘","’"), ("«","»"), ('"', '"')]
DASH_STYLES = ["—", "–", "--"]
ENDINGS = ["!", "?!", "!?", "‽", "!!", "!!!"]

def _is_word(tok: str) -> bool:
    return bool(_WORD_RE.fullmatch(tok))

def _tokenize(s: str) -> List[str]:
    # Keep punctuation tokens so we can preserve original punctuation.
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*|[^\sA-Za-z0-9]", s)

def _words_only(tokens: List[str]) -> List[str]:
    return [t for t in tokens if _is_word(t)]

def _find_pp_spans(tokens: List[str]) -> List[Tuple[int,int]]:
    """Find candidate prepositional phrase spans (prep ... until next hard punctuation)."""
    hard_punct = {",",";",";",":","?","!","(",")","[","]","—","–","—","--"}
    spans = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i].lower()
        if _is_word(tokens[i]) and t in PREP:
            j = i
            # extend while we haven't hit hard punctuation and still have words/apostrophes/hyphens
            while j + 1 < n and tokens[j+1] not in hard_punct:
                j += 1
                # stop if we run into sentence-final punctuation token
                if tokens[j] in {"?","!"}:
                    break
            if j > i:
                spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans

def _find_subclause_spans(tokens: List[str]) -> List[Tuple[int,int]]:
    """Find spans starting with that/which/who/because/if/when/unless ... up to next hard punctuation."""
    starters = {"that","which","who","because","if","when","unless","where","while"}
    hard_punct = {",",";",";",":","?","!","(",")","[","]","—","–","--"}
    spans = []
    n = len(tokens)
    for i,t in enumerate(tokens):
        if _is_word(t) and t.lower() in starters:
            j = i
            while j + 1 < n and tokens[j+1] not in hard_punct:
                j += 1
                if tokens[j] in {"?","!"}:
                    break
            if j > i:
                spans.append((i, j))
    return spans

def infect_punct_max_var(claim: str, intensity: int = 2, seed: int | None = None) -> str:
    """
    Heavy, variable, grammar-safe non-period punctuation infection.

    Guarantees:
      - Words and their order are unchanged.
      - Only non-period punctuation is added.
      - Avoids commas/colons/semicolons to keep grammar safe.
      - Randomized patterns to avoid being learned easily by a model.

    Parameters:
      intensity: 1 (light) .. 3 (max)
      seed: optional RNG seed for reproducibility
    """
    rng = random.Random(seed)

    # 1) Keep original tokens; drop only trailing periods (if any).
    s = claim.strip()
    s = re.sub(r"\.\s*$", "", s)
    tokens = _tokenize(s)
    n = len(tokens)

    # Prefix/suffix holders for every token (we attach punctuation here).
    pre = [""] * n
    post = [""] * n

    # 2) QUOTES around salient content words (1..k)
    word_idxs = [i for i,t in enumerate(tokens) if _is_word(t)]
    content = [i for i in word_idxs if len(tokens[i]) >= 4 and tokens[i].lower() not in STOP]
    rng.shuffle(content)
    k_quotes = max(1, min(intensity + 1, len(content)))  # 2–4 at max intensity depending on content
    for i in content[:k_quotes]:
        lq, rq = rng.choice(QUOTE_STYLES)
        pre[i] += lq
        post[i] = rq + post[i]

    # 3) Em/En dash around operators/negators (up to intensity+1 distinct hits)
    op_candidates = [i for i in word_idxs if tokens[i].lower() in OPERATORS]
    rng.shuffle(op_candidates)
    k_ops = min(len(op_candidates), intensity + 1)
    for i in op_candidates[:k_ops]:
        dash = rng.choice(DASH_STYLES)
        pre[i] += dash
        post[i] = dash + post[i]

    # 4) Dash around a couple of coordinators (and/or/but/nor)
    coord_candidates = [i for i in word_idxs if tokens[i].lower() in COORDS]
    rng.shuffle(coord_candidates)
    k_coord = min(len(coord_candidates), 1 + (intensity > 1))
    for i in coord_candidates[:k_coord]:
        dash = rng.choice(DASH_STYLES)
        pre[i] += dash
        post[i] = dash + post[i]

    # 5) Dash around some adverbs ending in -ly (optional)
    ly_candidates = [i for i in word_idxs if tokens[i].lower().endswith("ly") and tokens[i].lower() not in {"only"}]
    rng.shuffle(ly_candidates)
    k_ly = 1 if (intensity >= 2 and ly_candidates) else 0
    for i in ly_candidates[:k_ly]:
        dash = rng.choice(DASH_STYLES)
        pre[i] += dash
        post[i] = dash + post[i]

    # 6) Parenthetical/bracketed spans: choose from PPs and sub-clauses, avoid overlaps, random positions.
    spans = _find_pp_spans(tokens) + _find_subclause_spans(tokens)
    rng.shuffle(spans)
    chosen_spans = []
    used = set()
    target_spans = 1 + (intensity >= 2)  # 1–2 spans at higher intensity
    for (a,b) in spans:
        if any(a <= u <= b for u in used):
            continue
        chosen_spans.append((a,b))
        used.update(range(a,b+1))
        if len(chosen_spans) >= target_spans:
            break
    for (a,b) in chosen_spans:
        open_br, close_br = rng.choice([("(",")"), ("[","]")])
        pre[a] = open_br + pre[a]
        post[b] = post[b] + close_br

    # 7) Terminal punctuation variety (ensure no period)
    #    If claim already ends with ?/!, replace with a varied ending.
    ending = rng.choice(ENDINGS)

    # 8) Rebuild string with simple spacing rules.
    out = []
    for i, tok in enumerate(tokens):
        piece = pre[i] + tok + post[i]
        if i == 0:
            out.append(piece)
            continue
        prev = tokens[i-1]
        # spacing heuristic: no space before closing punct, or when previous ended with opening bracket/quote
        if tok in {",",";",";",":","?","!",")","]","’","”"}:
            # attach directly
            pass
        elif prev in {"(","[","‘","“"}:
            # attach directly
            pass
        else:
            out.append(" ")
        out.append(piece)
    s2 = "".join(out).strip()

    # strip existing terminal ?/! then add our chosen ending
    s2 = re.sub(r"[!?]+$", "", s2).rstrip() + ending

    # 9) Clean spacing around dashes/brackets/quotes
    s2 = re.sub(r"\s*([—–-]{2,})\s*", r"\1", s2)  # tight around dashes we inserted
    s2 = re.sub(r"\(\s+", "(", s2)
    s2 = re.sub(r"\s+\)", ")", s2)
    s2 = re.sub(r"\[\s+", "[", s2)
    s2 = re.sub(r"\s+\]", "]", s2)
    s2 = re.sub(r"\s+([!?])", r"\1", s2)

    # 10) Safety check: ensure word sequence preserved.
    orig_words = [w.lower() for w in _words_only(_tokenize(claim))]
    new_words  = [w.lower() for w in _words_only(_tokenize(s2))]
    if orig_words != new_words:
        # If something went off (rare), fall back to a safer minimal transform (quotes + ending).
        tokens = _tokenize(claim.strip())
        pre = [""] * len(tokens); post = [""] * len(tokens)
        widxs = [i for i,t in enumerate(tokens) if _is_word(t)]
        if widxs:
            i = widxs[0]
            lq,rq = "“","”"
            pre[i] += lq; post[i] = rq + post[i]
        s2 = "".join((pre[i] + tokens[i] + post[i] + (" " if i+1 < len(tokens) and _is_word(tokens[i]) and _is_word(tokens[i+1]) else ""))
                     for i in range(len(tokens))).strip()
        s2 = re.sub(r"\.\s*$", "", s2)
        s2 = re.sub(r"[!?]+$", "", s2) + "!"

    return s2

import re, spacy
from lemminflect import getInflection

# load once
_nlp = spacy.load("en_core_web_sm")

def _cap_like(src: str, dst: str) -> str:
    return dst.capitalize() if src and src[0].isupper() else dst

def _is_present(tok) -> bool:
    return tok.tag_ in {"VBZ","VBP"} or ("Pres" in tok.morph.get("Tense"))

def _is_past(tok) -> bool:
    return tok.tag_ == "VBD" or ("Past" in tok.morph.get("Tense"))

def _vb(lemma: str) -> str:
    inf = getInflection(lemma, tag="VB")
    return inf[0] if inf else lemma

def _vbd(lemma: str) -> str:
    inf = getInflection(lemma, tag="VBD")
    return inf[0] if inf else lemma

def _choose_was_were(verb_tok) -> str:
    # crude but effective: agree with nearest explicit subject if we can
    anchor = verb_tok.head if verb_tok.dep_ in ("aux","auxpass") else verb_tok
    subj = None
    for ch in anchor.children:
        if ch.dep_ in ("nsubj","nsubjpass","expl"):
            subj = ch; break
    if subj is None and verb_tok.dep_ in ("aux","auxpass"):
        for ch in anchor.head.children:
            if ch.dep_ in ("nsubj","nsubjpass","expl"):
                subj = ch; break
    if subj is not None:
        t = subj.text.lower()
        if t in {"we","they","you"}: return "were"
        if t == "i": return "was"
        if "Plur" in subj.morph.get("Number"): return "were"
    return "was"

def to_past_tense(text: str) -> str:
    doc = _nlp(text)
    out = []
    i, n = 0, len(doc)
    while i < n:
        tok = doc[i]; t, low = tok.text, tok.lower_

        # contracted negatives
        if i+1 < n and doc[i+1].text == "n't":
            nxt = doc[i+1]
            if low in {"wo","will"}: out += [_cap_like(t,"would"), " not", nxt.whitespace_]; i+=2; continue
            if low in {"ca","can"}:  out += [_cap_like(t,"could"), " not", nxt.whitespace_]; i+=2; continue
            if low in {"do","does"}: out += [_cap_like(t,"did"),   " not", nxt.whitespace_]; i+=2; continue
            if low in {"is","are"}: out += [_cap_like(t,_choose_was_were(tok)), " not", nxt.whitespace_]; i+=2; continue
            if low in {"has","have"}: out += [_cap_like(t,"had"), " not", nxt.whitespace_]; i+=2; continue
            out += [t, " not", nxt.whitespace_]; i+=2; continue

        # modals: will→would; shall→should; can→could; may→might
        if tok.tag_ == "MD":
            mapping = {"will":"would","wo":"would","shall":"should","can":"could","ca":"could","may":"might"}
            if low in mapping:
                out += [_cap_like(t, mapping[low]), tok.whitespace_]; i+=1; continue

        # be/have/do (present) → past
        if tok.lemma_ == "be" and _is_present(tok):
            ww = _choose_was_were(tok)
            out += [_cap_like(t, ww), tok.whitespace_]; i+=1; continue
        if tok.lemma_ == "have" and _is_present(tok):
            out += [_cap_like(t, "had"), tok.whitespace_]; i+=1; continue
        if tok.lemma_ == "do" and _is_present(tok):
            out += [_cap_like(t, "did"), tok.whitespace_]; i+=1; continue

        # finite present verb → VBD
        if tok.pos_ in {"VERB","AUX"} and _is_present(tok) and tok.tag_ not in {"VBG","VBN"}:
            out += [_cap_like(t, _vbd(tok.lemma_)), tok.whitespace_]; i+=1; continue

        # everything else
        out += [t, tok.whitespace_]; i+=1

    s = "".join(out)
    s = re.sub(r"\b[Cc]annot\b", lambda m: _cap_like(m.group(0), "could not"), s)
    return s

def to_future_tense(text: str) -> str:
    doc = _nlp(text)
    out = []
    i, n = 0, len(doc)
    while i < n:
        tok = doc[i]; t, low = tok.text, tok.lower_

        # contracted negatives
        if i+1 < n and doc[i+1].text == "n't":
            nxt = doc[i+1]
            if low in {"wo","will"}: out += [_cap_like(t,"will"), " not", nxt.whitespace_]; i+=2; continue
            if low in {"ca","can","could"}: out += [_cap_like(t,"will"), " not be able to", nxt.whitespace_]; i+=2; continue
            if low in {"do","does","did"}: out += [_cap_like(t,"will"), " not", nxt.whitespace_]; i+=2; continue
            if low in {"is","are","was","were","be"}: out += [_cap_like(t,"will"), " not be", nxt.whitespace_]; i+=2; continue
            if low in {"has","have","had"}: out += [_cap_like(t,"will"), " not have", nxt.whitespace_]; i+=2; continue
            out += [t, " not", nxt.whitespace_]; i+=2; continue

        # modals → normalize to will / will be able to
        if tok.tag_ == "MD":
            if low in {"will","wo","would","shall"}:
                out += [_cap_like(t,"will"), tok.whitespace_]; i+=1; continue
            if low in {"can","ca","could"}:
                out += [_cap_like(t,"will be able to"), tok.whitespace_]; i+=1; continue
            if low in {"may","might"}:
                out += [_cap_like(t,"will"), tok.whitespace_]; i+=1; continue
            out += [t, tok.whitespace_]; i+=1; continue

        # BE finite (present or past) → will be
        if tok.lemma_ == "be" and (_is_present(tok) or _is_past(tok)):
            out += [_cap_like(t,"will be"), tok.whitespace_]; i+=1; continue

        # HAVE finite → will have
        if tok.lemma_ == "have" and (_is_present(tok) or _is_past(tok)):
            out += [_cap_like(t,"will have"), tok.whitespace_]; i+=1; continue

        # DO finite → will
        if tok.lemma_ == "do" and (_is_present(tok) or _is_past(tok)):
            out += [_cap_like(t,"will"), tok.whitespace_]; i+=1; continue

        # other finite verbs (present/past) → will + base (skip participles)
        if tok.pos_ in {"VERB","AUX"} and ( _is_present(tok) or _is_past(tok) ) and tok.tag_ not in {"VBG","VBN"}:
            out += [_cap_like(t, "will " + _vb(tok.lemma_)), tok.whitespace_]; i+=1; continue

        out += [t, tok.whitespace_]; i+=1

    s = "".join(out)
    # missed "cannot"
    s = re.sub(r"\b[Cc]annot\b", lambda m: _cap_like(m.group(0), "will not be able to"), s)
    # collapse any double spaces
    s = re.sub(r"[ ]{2,}", " ", s)
    return s

def get_predictions(client, MODEL_PATH, positive_prompts, negative_prompts, batch_size = 32):

    # Combine positive and negative prompts for batch processing
    all_prompts = positive_prompts + negative_prompts

    print(f"Processing {len(all_prompts)} prompts...")

    # Process prompts in batches
    
    all_true_logprobs = []
    all_false_logprobs = []
    all_preds = []  # 1 => "True", 0 => "False" (from generated token if possible, else margin)

    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        # print(f"Processing batch {i//batch_size + 1}/{(len(all_prompts)-1)//batch_size + 1}")
        
        # Get completions using the OpenAI client
        batch_true_logprobs = []
        batch_false_logprobs = []
        batch_preds = []
        
        for prompt in batch_prompts:
            try:
                response = client.completions.create(
                    model=MODEL_PATH,  # Use the model path from your config
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=20, # Get more logprobs to ensure we capture True/False
                    echo=False  # Don't include prompt tokens in response
                )
                
                # Extract log probs for True and False tokens
                true_logprob = None
                false_logprob = None
                
                if response.choices[0].logprobs and response.choices[0].logprobs.top_logprobs:
                    # Get the logprobs dictionary for the first (and only) generated position
                    top_logprobs = response.choices[0].logprobs.top_logprobs[0]
                    
                    # Look for True and False tokens in the logprobs
                    for token, logprob in top_logprobs.items():
                        if token == " True":
                            true_logprob = logprob
                        elif token == " False":
                            false_logprob = logprob
                
                # If we didn't find the tokens in top logprobs, assign very low probability
                if true_logprob is None:
                    true_logprob = -20.0  # Very low log probability
                if false_logprob is None:
                    false_logprob = -20.0  # Very low log probability

                # Determine predicted token:
                # Prefer the actual generated token if it is clearly True/False;
                # otherwise fall back to margin sign.
                gen_text = getattr(response.choices[0], "text", "")
                # s = gen_text.strip()
                if gen_text.startswith(" True"):
                    pred = 1
                elif gen_text.startswith(" False"):
                    pred = 0
                else:
                    pred = int(true_logprob >= false_logprob)
                    
                batch_true_logprobs.append(true_logprob)
                batch_false_logprobs.append(false_logprob)
                batch_preds.append(pred)
                    
            except Exception as e:
                print(f"Error processing prompt: {e}")
                batch_true_logprobs.append(-20.0)
                batch_false_logprobs.append(-20.0)
                batch_preds.append(0)  # arbitrary fallback

        all_true_logprobs.extend(batch_true_logprobs)
        all_false_logprobs.extend(batch_false_logprobs)
        all_preds.extend(batch_preds)

    # Convert to numpy arrays for easier manipulation
    all_true_logprobs = np.array(all_true_logprobs)
    all_false_logprobs = np.array(all_false_logprobs)
    all_preds = np.array(all_preds, dtype=int)

    # Split back into positive and negative log probs / preds
    positive_true_logprobs  = all_true_logprobs[:len(positive_prompts)]
    positive_false_logprobs = all_false_logprobs[:len(positive_prompts)]
    negative_true_logprobs  = all_true_logprobs[len(positive_prompts):]
    negative_false_logprobs = all_false_logprobs[len(positive_prompts):]

    positive_preds = all_preds[:len(positive_prompts)]
    negative_preds = all_preds[len(positive_prompts):]

    return (
        positive_true_logprobs,
        positive_false_logprobs,
        negative_true_logprobs,
        negative_false_logprobs,
        positive_preds,
        negative_preds,
    )

def get_direct_and_consistent_predictions(
    client,
    MODEL_PATH,
    positive_prompts: list[str],
    negative_prompts: list[str],
    batch_size: int = 32,
):
    """
    Compute direct and pairwise-consistent predictions for given positive/negative prompts.

    - Direct: for each prompt independently, pred=True iff logP(True) - logP(False) >= 0
    - Consistent: for each (pos_i, neg_i) pair, assign True to the side with larger margin

    Returns two tuples of numpy int arrays (each of shape (P,)):
      (direct_pos_preds, direct_neg_preds), (consist_pos_preds, consist_neg_preds)
    """

    all_prompts = list(positive_prompts) + list(negative_prompts)
    print(f"Processing {len(all_prompts)} prompts...")

    all_true_logprobs: list[float] = []
    all_false_logprobs: list[float] = []

    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        batch_true_logprobs: list[float] = []
        batch_false_logprobs: list[float] = []

        for prompt in batch_prompts:
            try:
                response = client.completions.create(
                    model=MODEL_PATH,
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=20,
                    echo=False,
                )
                true_logprob = None
                false_logprob = None
                if response.choices[0].logprobs and response.choices[0].logprobs.top_logprobs:
                    top_logprobs = response.choices[0].logprobs.top_logprobs[0]
                    for token, logprob in top_logprobs.items():
                        if token == " True":
                            true_logprob = logprob
                        elif token == " False":
                            false_logprob = logprob
                if true_logprob is None:
                    true_logprob = -20.0
                if false_logprob is None:
                    false_logprob = -20.0
                batch_true_logprobs.append(true_logprob)
                batch_false_logprobs.append(false_logprob)
            except Exception as e:
                print(f"Error processing prompt: {e}")
                batch_true_logprobs.append(-20.0)
                batch_false_logprobs.append(-20.0)

        all_true_logprobs.extend(batch_true_logprobs)
        all_false_logprobs.extend(batch_false_logprobs)

    all_true_logprobs = np.array(all_true_logprobs, dtype=float)
    all_false_logprobs = np.array(all_false_logprobs, dtype=float)

    P = len(positive_prompts)
    pos_true = all_true_logprobs[:P]
    pos_false = all_false_logprobs[:P]
    neg_true = all_true_logprobs[P:]
    neg_false = all_false_logprobs[P:]

    m_pos = pos_true - pos_false
    m_neg = neg_true - neg_false

    # Direct predictions: True if margin >= 0
    direct_pos = (m_pos >= 0).astype(int)
    direct_neg = (m_neg >= 0).astype(int)

    # Consistent (pairwise) predictions: True to side with strictly larger margin
    consist_pos = (m_pos > m_neg).astype(int)
    consist_neg = 1 - consist_pos

    return (direct_pos, direct_neg), (consist_pos, consist_neg)

def direct_accuracy_from_splits(
    pos_true_logprobs, pos_false_logprobs,
    neg_true_logprobs, neg_false_logprobs,
    y_truth,
    tie_policy="true_on_tie"  # {"true_on_tie", "false_on_tie", "skip"}
):
    p_t = np.array([np.nan if v is None else v for v in pos_true_logprobs], dtype=float)
    p_f = np.array([np.nan if v is None else v for v in pos_false_logprobs], dtype=float)
    n_t = np.array([np.nan if v is None else v for v in neg_true_logprobs], dtype=float)
    n_f = np.array([np.nan if v is None else v for v in neg_false_logprobs], dtype=float)
    y   = np.asarray(y_truth, dtype=int)

    m_pos = p_t - p_f
    m_neg = n_t - n_f

    valid_pos = ~np.isnan(m_pos)
    valid_neg = ~np.isnan(m_neg)

    pred_pos = np.zeros_like(y)
    pred_neg = np.zeros_like(y)

    pred_pos[valid_pos] = (m_pos[valid_pos] > 0).astype(int)
    pred_neg[valid_neg] = (m_neg[valid_neg] > 0).astype(int)

    ties_pos = valid_pos & (m_pos == 0)
    ties_neg = valid_neg & (m_neg == 0)
    if tie_policy == "true_on_tie":
        pred_pos[ties_pos] = 1
        pred_neg[ties_neg] = 1
    elif tie_policy == "false_on_tie":
        pred_pos[ties_pos] = 0
        pred_neg[ties_neg] = 0
    elif tie_policy == "skip":
        valid_pos = valid_pos & (m_pos != 0)
        valid_neg = valid_neg & (m_neg != 0)
    else:
        raise ValueError("tie_policy must be one of {'true_on_tie','false_on_tie','skip'}")

    gt_pos = y
    gt_neg = 1 - y

    correct = int((pred_pos[valid_pos] == gt_pos[valid_pos]).sum()
                  + (pred_neg[valid_neg] == gt_neg[valid_neg]).sum())
    total   = int(valid_pos.sum() + valid_neg.sum())
    skipped = int(2*len(y) - total)

    acc = correct / total if total > 0 else float("nan")
    return acc, {
        "correct_items": correct,
        "total_items": total,
        "skipped_items": skipped,
        "ties_pos": int(ties_pos.sum()),
        "ties_neg": int(ties_neg.sum()),
    }


# ========== CONSISTENT ACCURACY (pairwise) ==========
# For each pair, compute margins (pos and neg). Assign True to the side with the
# larger margin and False to the other. Score both items per pair.
def consistent_accuracy_from_splits(
    pos_true_logprobs, pos_false_logprobs,
    neg_true_logprobs, neg_false_logprobs,
    y_truth,
    tie_policy="pos_true_on_tie"  # {"pos_true_on_tie","pos_false_on_tie","skip"}
):
    p_t = np.array([np.nan if v is None else v for v in pos_true_logprobs], dtype=float)
    p_f = np.array([np.nan if v is None else v for v in pos_false_logprobs], dtype=float)
    n_t = np.array([np.nan if v is None else v for v in neg_true_logprobs], dtype=float)
    n_f = np.array([np.nan if v is None else v for v in neg_false_logprobs], dtype=float)
    y   = np.asarray(y_truth, dtype=int)

    m_pos = p_t - p_f
    m_neg = n_t - n_f
    valid_pairs = ~np.isnan(m_pos) & ~np.isnan(m_neg)

    # Which side gets True?
    pred_pos_true = np.zeros_like(y, dtype=bool)
    pred_pos_true[valid_pairs] = (m_pos[valid_pairs] > m_neg[valid_pairs])

    ties = valid_pairs & (m_pos == m_neg)
    if tie_policy == "pos_true_on_tie":
        pred_pos_true[ties] = True
    elif tie_policy == "pos_false_on_tie":
        pred_pos_true[ties] = False
    elif tie_policy == "skip":
        valid_pairs = valid_pairs & (m_pos != m_neg)
    else:
        raise ValueError("tie_policy must be one of {'pos_true_on_tie','pos_false_on_tie','skip'}")

    pred_neg_true = ~pred_pos_true

    gt_pos_true = (y == 1)
    gt_neg_true = ~gt_pos_true

    pairs_scored   = int(valid_pairs.sum())
    correct_items  = int((pred_pos_true[valid_pairs] == gt_pos_true[valid_pairs]).sum()
                         + (pred_neg_true[valid_pairs] == gt_neg_true[valid_pairs]).sum())
    total_items    = pairs_scored * 2
    skipped_pairs  = int(len(y) - pairs_scored)

    acc_items = correct_items / total_items if total_items > 0 else float("nan")
    return acc_items, {
        "pairs_scored": pairs_scored,
        "correct_items": correct_items,
        "total_items": total_items,
        "skipped_pairs": skipped_pairs,
        "ties": int(ties.sum()),
    }


def accuracy_from_predictions(y, pos_preds, neg_preds):
    """
    Compute item-level accuracy given per-pair labels and predicted booleans.

    Args:
      y_truth: array-like of shape (P,), 1 if positive side should be True, 0 otherwise
      pos_preds: array-like of shape (P,), predicted 1/0 for positive prompts
      neg_preds: array-like of shape (P,), predicted 1/0 for negative prompts

    Returns:
      float accuracy over 2P items: (pos==y) + (neg==1-y)) / (2P)
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(pos_preds, dtype=int)
    n = np.asarray(neg_preds, dtype=int)
    assert y.shape == p.shape == n.shape, "y_truth, pos_preds, neg_preds must have same shape"
    correct = int((p == y).sum() + (n == (1 - y)).sum())
    total = 2 * len(y)
    return correct / total if total > 0 else float("nan")
