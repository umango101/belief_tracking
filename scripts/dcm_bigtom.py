import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import argparse

sys.path.append("../")
from utils import *

from nnsight import LanguageModel

# Set random seed for reproducibility
seed = 10
np.random.seed(seed)
torch.manual_seed(seed)

# Define helper functions
def get_ques_start_token_idx(batch_size, tokenizer, prompt, padding_side="right"):
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True, padding_side=padding_side).input_ids
    colon_token = tokenizer.encode(":", return_tensors="pt").squeeze()[-1].item()
    ques_start_idx = (input_tokens == colon_token).nonzero()[torch.arange(2, 4*batch_size, 4)][:, 1] - 1
    return ques_start_idx

def get_prompt_token_len(tokenizer, prompt, padding_side="right"):
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True, padding_side=padding_side)
    return input_tokens.attention_mask.sum(dim=-1)

def train_mask(model, dataset_func, layer_range, lambs, n_epochs=2, train_size=80, valid_size=40, test_size=80, variable_name='answer', batch_size=1):
    """Train masks for specified layers using the given dataset function"""
    
    # Load dataset
    df_false = pd.read_csv("../data/bigtom/0_forward_belief_false_belief/stories.csv", delimiter=";")
    df_true = pd.read_csv("../data/bigtom/0_forward_belief_true_belief/stories.csv", delimiter=";")
    
    dataset = dataset_func(df_false, df_true, train_size+valid_size+test_size)
    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size:train_size+valid_size]
    test_dataset = dataset[train_size+valid_size:]
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load singular vectors for each layer
    sing_vecs = defaultdict(dict)
    for l in range(model.config.num_hidden_layers):
        if dataset_func.__name__ == "get_bigtom_value_fetcher_exps":
            sing_vecs[l] = torch.load(f"../svd_results/BigToM/last_token/singular_vecs/{l}.pt").cpu()
        elif dataset_func.__name__ == "get_bigtom_answer_state_exps":
            sing_vecs[l] = torch.load(f"../svd_results/BigToM/last_token/singular_vecs/{l}.pt").cpu()
        elif dataset_func.__name__ == "get_bigtom_query_charac":
            sing_vecs[l] = torch.load(f"../svd_results/BigToM/query_charac_new/singular_vecs/{l}.pt").cpu()
    
    valid_accs = defaultdict(dict)
    
    # Set padding side for tokenization
    model.tokenizer.padding_side = "left"
    
    # Train masks for each layer and regularization parameter
    for layer_idx in layer_range:
        for lamb in lambs:
            print(f"Training layer: {layer_idx}, lambda: {lamb}")
            
            # Initialize mask
            modules = [i for i in range(sing_vecs[layer_idx].size(0))]
            mask = torch.ones(len(modules), requires_grad=True, device="cuda", dtype=torch.bfloat16)
            optimizer = torch.optim.Adam([mask], lr=1e-1)
            
            # Training loop
            for epoch in range(n_epochs):
                epoch_loss = 0
                
                for bi, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                    alt_prompt = batch["alt_prompt"]
                    org_prompt = batch["org_prompt"]
                    target = batch["target"]
                    target_token = model.tokenizer(target, return_tensors="pt", padding=True, padding_side="right")
                    target_input_ids = target_token.input_ids[:, 1:]
                    batch_size = target_input_ids.size(0)
                    
                    alt_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, alt_prompt, padding_side="right")
                    alt_prompt_len = get_prompt_token_len(model.tokenizer, alt_prompt, padding_side="right")
                    org_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, org_prompt, padding_side="right")
                    org_prompt_len = get_prompt_token_len(model.tokenizer, org_prompt, padding_side="right")
                    
                    optimizer.zero_grad()
                    
                    with model.trace() as tracer:
                        # Different handling based on the dataset type
                        if dataset_func.__name__ in ["get_bigtom_value_fetcher_exps", "get_bigtom_answer_state_exps"]:
                            # For Answer Variable and Answer State OID Variable
                            with tracer.invoke(alt_prompt):
                                alt_acts = model.model.layers[layer_idx].output[0][0, -1].clone().save()
                            
                            with tracer.invoke(org_prompt):
                                sing_vec = sing_vecs[layer_idx].cuda()
                                masked_vec = sing_vec * mask.unsqueeze(-1)
                                proj_matrix = torch.matmul(masked_vec.t(), masked_vec).half()
                                
                                curr_output = model.model.layers[layer_idx].output[0][0, -1].clone()
                                
                                alt_proj = torch.matmul(alt_acts, proj_matrix)
                                org_proj = torch.matmul(curr_output, proj_matrix)
                                
                                modified_out = curr_output - org_proj + alt_proj
                                model.model.layers[layer_idx].output[0][0, -1] = modified_out
                                
                                logits = model.lm_head.output[0, -1].save()
                                
                                del sing_vec, proj_matrix, masked_vec
                                torch.cuda.empty_cache()
                                
                        else:
                            # For Query Character Variable
                            alt_acts = defaultdict(dict)
                            with tracer.invoke(alt_prompt):
                                for t_idx, t in enumerate([i for i in range(alt_ques_idx+3, alt_ques_idx+5)]):
                                    alt_acts[t_idx] = model.model.layers[layer_idx].output[0][0, t].clone().save()
                            
                            with tracer.invoke(org_prompt):
                                sing_vec = sing_vecs[layer_idx].cuda()
                                masked_vec = sing_vec * mask.unsqueeze(-1)
                                proj_matrix = torch.matmul(masked_vec.t(), masked_vec).half()
                                
                                for t_idx, t in enumerate([i for i in range(org_ques_idx+3, org_ques_idx+5)]):
                                    curr_output = model.model.layers[layer_idx].output[0][0, t].clone()
                                    
                                    alt_proj = torch.matmul(alt_acts[t_idx], proj_matrix)
                                    org_proj = torch.matmul(curr_output, proj_matrix)
                                    
                                    modified_out = curr_output - org_proj + alt_proj
                                    model.model.layers[layer_idx].output[0][0, t] = modified_out
                                
                                logits = model.lm_head.output[0, -1].save()
                                
                                del sing_vec, proj_matrix, masked_vec
                                torch.cuda.empty_cache()
                    
                    target_logit = logits[target_input_ids[0]].sum()
                    
                    task_loss = -(target_logit/batch_size)
                    l1_loss = lamb * torch.norm(mask, p=1)
                    loss = task_loss + l1_loss.to(task_loss.device)
                    
                    epoch_loss += loss.item()
                    
                    if bi % 4 == 0:
                        mean_loss = epoch_loss / (bi + 1)
                        print(f"Epoch: {epoch}, Batch: {bi}, Task Loss: {task_loss.item():.4f}, "
                              f"L1 Loss: {l1_loss.item():.4f}, Total Loss: {mean_loss:.4f}")
                        with torch.no_grad():
                            mask.data.clamp_(0, 1)
                            rounded = torch.round(mask)
                            print(f"#Rank: {(rounded == 1).sum().item()}")
                    
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        mask.data.clamp_(0, 1)
            
            print(f"Training finished for layer: {layer_idx}, lambda: {lamb}")
            
            # Validation
            print(f"Validation started for layer: {layer_idx}, lambda: {lamb}")
            correct, total = 0, 0
            
            with torch.inference_mode():
                mask_data = mask.data.clone()
                mask_data.clamp_(0, 1)
                rounded = torch.round(mask)
                
                print(f"Rank: {(rounded == 1).sum().item()}")
                
                # Save the mask
                os.makedirs(f"../masks/BigToM/{variable_name}", exist_ok=True)
                mask_name = f"{layer_idx}.pt"
                torch.save(mask_data, f"../masks/BigToM/{variable_name}/{mask_name}")
                
                for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    alt_prompt = batch["alt_prompt"]
                    org_prompt = batch["org_prompt"]
                    alt_ans = batch["alt_ans"]
                    target = batch["target"][0]
                    batch_size = len(alt_ans)
                    
                    alt_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, alt_prompt, padding_side="left")
                    alt_prompt_len = get_prompt_token_len(model.tokenizer, alt_prompt, padding_side="left")
                    org_ques_idx = get_ques_start_token_idx(batch_size, model.tokenizer, org_prompt, padding_side="left")
                    org_prompt_len = get_prompt_token_len(model.tokenizer, org_prompt, padding_side="left")
                    
                    with model.session() as session:
                        # Different handling based on the dataset type
                        if dataset_func.__name__ in ["get_bigtom_value_fetcher_exps", "get_bigtom_answer_state_exps"]:
                            # For Answer Variable and Answer State OID Variable
                            with model.trace(alt_prompt):
                                alt_acts = model.model.layers[layer_idx].output[0][0, -1].save()
                            
                            with model.generate(org_prompt, max_new_tokens=2, do_sample=False, num_return_sequences=1, 
                                              pad_token_id=model.tokenizer.pad_token_id, eos_token_id=model.tokenizer.eos_token_id):
                                sing_vec = sing_vecs[layer_idx].cuda()
                                masked_vec = sing_vec * rounded.unsqueeze(-1)
                                proj_matrix = torch.matmul(masked_vec.t(), masked_vec).half()
                                
                                curr_output = model.model.layers[layer_idx].output[0][0, -1].clone()
                                
                                alt_proj = torch.matmul(alt_acts, proj_matrix)
                                org_proj = torch.matmul(curr_output, proj_matrix)
                                
                                modified_out = curr_output - org_proj + alt_proj
                                model.model.layers[layer_idx].output[0][0, -1] = modified_out
                                
                                out = model.generator.output.save()
                                
                            del alt_acts
                            torch.cuda.empty_cache()
                            
                        else:
                            # For Query Character Variable
                            alt_layer_out = defaultdict(dict)
                            with model.trace(alt_prompt):
                                for t_idx, t in enumerate([i for i in range(alt_ques_idx+3, alt_ques_idx+5)]):
                                    alt_layer_out[t_idx] = model.model.layers[layer_idx].output[0][0, t].save()
                            
                            with model.generate(org_prompt, max_new_tokens=2, do_sample=False, num_return_sequences=1, 
                                              pad_token_id=model.tokenizer.pad_token_id, eos_token_id=model.tokenizer.eos_token_id):
                                sing_vec = sing_vecs[layer_idx].cuda()
                                masked_vec = sing_vec * rounded.unsqueeze(-1)
                                proj_matrix = torch.matmul(masked_vec.t(), masked_vec).half()
                                
                                for t_idx, t in enumerate([i for i in range(org_ques_idx+3, org_ques_idx+5)]):
                                    curr_output = model.model.layers[layer_idx].output[0][0, t].clone()
                                    alt_proj = torch.matmul(alt_layer_out[t_idx], proj_matrix)
                                    org_proj = torch.matmul(curr_output, proj_matrix)
                                    modified_out = curr_output - org_proj + alt_proj
                                    model.model.layers[layer_idx].output[0][0, t] = modified_out
                                
                                out = model.generator.output.save()
                                
                            del alt_layer_out
                            torch.cuda.empty_cache()
                    
                    pred = model.tokenizer.decode(out[0][org_prompt_len:-1]).strip()
                    # print(f"Prediction: {pred} | Target: {target}")
                    if pred.lower() in target.lower():
                        correct += 1
                    total += 1
                
                print(f"Validation accuracy: {correct / total:.2f} | Correct: {correct} | Total: {total}\n")
                valid_accs[lamb][layer_idx] = round(correct / total, 2)
    
    return valid_accs

def main():
    parser = argparse.ArgumentParser(description="Train masks for ToM variables")
    parser.add_argument('--variable', type=str, required=True, choices=['answer', 'answer_state', 'query_character'],
                        help='Which variable to train: answer, answer_state, or query_character')
    parser.add_argument('--layer_start', type=int, default=0, help='Starting layer index')
    parser.add_argument('--layer_end', type=int, default=80, help='Ending layer index (exclusive)')
    parser.add_argument('--layer_step', type=int, default=2, help='Step between layers')
    parser.add_argument('--lambda_values', type=float, nargs='+', default=[0.05, 0.005], help='Regularization parameters')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--cache_dir', type=str, default="/disk/u/nikhil/.cache/huggingface/hub/", help='Cache directory for huggingface')
    args = parser.parse_args()

    # Print arguments
    print(f"Arguments: {args}")
    print(f"Variable: {args.variable}, Layer Range: {args.layer_start}-{args.layer_end}, "
            f"Lambda Values: {args.lambda_values}, Epochs: {args.epochs}")

    # Load the model
    model = LanguageModel("meta-llama/Meta-Llama-3-70B-Instruct", 
                          cache_dir=args.cache_dir, 
                          device_map="auto", 
                          load_in_4bit=True, 
                          torch_dtype=torch.float16, 
                          dispatch=True)
    
    # Set model to evaluation mode
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Set up layer range
    layer_range = range(args.layer_start, args.layer_end, args.layer_step)
    
    # Select dataset function based on variable
    if args.variable == 'answer':
        dataset_func = get_bigtom_value_fetcher_exps
    elif args.variable == 'answer_state':
        dataset_func = get_bigtom_answer_state_exps
    else:  # query_character
        dataset_func = get_bigtom_query_charac
    
    # Train masks
    results = train_mask(model, dataset_func, layer_range, args.lambda_values, n_epochs=args.epochs, variable_name=args.variable)
    
    # Save results
    results_df = pd.DataFrame.from_dict({(i, j): results[i][j] 
                                         for i in results.keys() 
                                         for j in results[i].keys()}, 
                                        orient='index')
    results_df.to_csv(f"bigtom_{args.variable}_results.csv")
    
    print("Complete! Results saved to CSV.")

if __name__ == "__main__":
    main()