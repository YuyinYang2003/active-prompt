# This file used to generate uncertainty score for each question
from utils import *
import time
import argparse
import numpy as np
import json
from scipy.stats import entropy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import sys
import time
from pathlib import Path
from tqdm import tqdm
import pdb

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.validation_set_type == "split-training-set":
        prompt=create_valset_loader(args)
        path = f"{args.prompt_path}"
        with open(path, 'w') as f:
            try:
                json.dump(prompt,f)
            except:
                pass
        
    set_random_seed(args.random_seed)

    dataloader = create_dataloader(args)

    if args.dataset_size > args.qes_limit:
        dataloader = dataloader[:args.qes_limit] # only take qes_limit questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")


    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)

    start =time.time()
    result=[]
    for i in range(len(dataloader)//args.qes_pair_num):
        result+=create_logdifference(args, dataloader[i*args.qes_pair_num:(i+1)*args.qes_pair_num], result)   
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

    # output the results
    path = f"{args.output_dir}/logdifference_result_{args.dataset}_from_{args.qes_limit}_questions.txt"
    with open(path, 'w') as f:
        try:
            f.write(json.dumps(result, indent=4))
        except:
            pass


def generate_logprob_qes(args, qes, model, tokenizer, qa_pairs, with_validation: bool):
    '''返回 logprob 标量值;
    TODO: 检查 output_path, 并把先前已经选出的 (x_0, y_0) 作为 context 接入 prompt.'''
    #qa_pairs是已经选出来的对
    if with_validation:
        prompt_text = create_input_prompt(args, qa_pairs, val_flag=True)
        prompt_text += qes["question"] + "\nA: " + qes["answer"]    # No `.` after answer
        logprob = calculate_logprob_ans(model, tokenizer, prompt_text)
    else:
        prompt_text = create_input_prompt(args, qa_pairs, val_flag=False)
        prompt_text += qes["question"] + "\nA: " + qes["answer"]
        logprob = calculate_logprob_ans(model, tokenizer, prompt_text)
    
    return logprob


def calculate_logprob_ans(model, tokenizer, input_prompt):
    '''Given model object, tokenizer object and prompt, return log-probability of the answer in prompt.'''
    encodings = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    input_ids = encodings["input_ids"]
    assert input_ids.numel() <= 4096, "input_prompt is too long for the model, should discard some contexts."
    labels = input_ids
    
    with torch.no_grad():
        out_logits = model(input_ids).logits
    
    answer_logit = out_logits[:, -2, :] # For prediction scores, answer is at position -2
    answer_label = labels[:, -1]    # For input_ids, answer is at position -1
    loss = torch.nn.CrossEntropyLoss()
    log_prob = - loss(answer_logit, answer_label)
    
    return log_prob


def create_logdifference(args, questions, qa_pairs):
    '''The argument provided for `questions` is `dataloader`'''
    result = []
    model_path = args.model
    dataset_length=len(questions)
    #if args.qes_limit > 0:
    #    questions = questions[:args.qes_limit]
    
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto") # 已经自动做了 device_map, 那就不需要 .to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)    # model_max_length=sys.maxsize

    for qes in tqdm(questions):
        logprob_with_validation = generate_logprob_qes(args, qes=qes, model=model, tokenizer=tokenizer, qa_pairs=qa_pairs, with_validation=True)
        logprob = generate_logprob_qes(args, qes=qes, model=model, tokenizer=tokenizer, qa_pairs=[], with_validation=False)
        log_difference = (logprob_with_validation - logprob).item()
        result.append({
            "dataset_idx": qes["question_idx"],
            "question": qes["question"],
            "answer": qes["answer"],
            "log_difference": log_difference
        })
    
    # Now sort the results by log_difference from big to small
    result.sort(key=lambda x: -x["log_difference"])
    if args.qes_limit>0:
        return result[:args.qes_each_time]
    else:
        return result[:5]


def arg_parser():
    parser = argparse.ArgumentParser(description="logdifference_calculation_and_sort")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k"], help="dataset to inference"
    )   # choices=["gsm8k","svamp", "aqua", "csqa", "last_letters", "strategyqa", "asdiv", "singleeq", "addsub", "multiarith"]
    parser.add_argument(
        "--model", type=str, default="baichuan-inc/Baichuan-7B", help="HuggingFace model used to calculate logprob"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./logdifference_results", help="output directory for logdifference results"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=7470, help="the total number of qa pairs from the training dataset to choose from"
    )
    parser.add_argument(
        "--qes_each_time", type=int, default=1, help="the number of qa pair we choose each time"
    )
    parser.add_argument(
        "--qes_pair_num", type=int, default=1494, help="the number of qa pairs we choose from each time"
    )
    parser.add_argument(
        "--valset_num", type=int, default=8, help="the number of qa pairs in the initial validation set"
    )
    parser.add_argument(
        "--validation_set_type", type=str, default="split-training-set", choices=["non-task-specific", "split-training-set"],help="the validation set we choose"
    )
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/train.jsonl" # train data path
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/train.json" # train data path
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/train_rand_split.jsonl" # train data path
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/train.json" # train data path
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_train2.json" # train data path
    else:
        raise ValueError("dataset is not properly defined ...")
    
    if args.validation_set_type == "non-task-specific":
        args.prompt_path = "./validation_prompts/math_word_problems" # val set data path
    elif args.validation_set_type == "split-training-set":
        args.prompt_path = "./validation_prompts/training_set_split" # val set data path
    
    return args


if __name__ == "__main__":
    main()
