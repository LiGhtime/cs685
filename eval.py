import json
import os
from datetime import datetime

from tqdm import tqdm
from datasets import load_from_disk
from transformers import GenerationConfig
import torch
from unsloth import FastLanguageModel

# define hyper_params
hyper_params = {
    "max_seq_length": 4096, # 8192 | Choose any! We auto support RoPE Scaling internally!
    "dtype": None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    "load_in_4bit": True, # Use 4bit quantization to reduce memory usage. Can be False.
    "json_file_asins": './data/asins_small.json', # './data/asins_small.json' or './data/meta_asins.json'
    "eval_dataset_path": "./data/gemma_chat_eval",
    # "model_name": "unsloth/gemma-2b-it-bnb-4bit",
    # "model_name": "outputs/checkpoint-1000",
    "model_name": "outputs/model_04242024_090830/",
    # "model_name": "outputs/model_04242024_150533",
    # "model_name": "outputs/model_04252024_034847",
    "early_stopping": True,
    "num_beams_parameter": 5,
    "max_new_tokens": 35,
    "temperature": 1,
    "use_cache": True,
    "num_beam_groups": 5,
    "diversity_penalty": 0.9,
}

# asins_small.json to be used for meta data matching
# json_file_asins = 
json_file_asins = hyper_params["json_file_asins"]
with open(json_file_asins, "r") as file:
    asin_dict = json.load(file)
    
# len(asin_dict) # 748224 for the full meta bag # 434236 after filtering out the ones with no title
    
# read eval dataset from local
dataset_eval = load_from_disk(hyper_params["eval_dataset_path"])
print("Loaded eval dataset from {}".format(hyper_params["eval_dataset_path"]))
# # take the first 10 rows just for testing
# dataset_eval = dataset_eval.select(range(2))
        
prompt_template = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n{}"

# initialize the results dict
results = {}

# load the model
max_seq_length = hyper_params["max_seq_length"]
dtype = hyper_params["dtype"]
load_in_4bit = hyper_params["load_in_4bit"]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = hyper_params["model_name"],
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# define the generation config
custom_generation_config = GenerationConfig(
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    asin_dict=asin_dict,
    tokenizer=tokenizer,
    return_dict_in_generate=True,
    output_scores=True,
    # output_logits=True,
    # do_sample=True,
    early_stopping=hyper_params["early_stopping"],
    num_beams=hyper_params["num_beams_parameter"], 
    num_return_sequences=hyper_params["num_beams_parameter"],
    max_new_tokens=hyper_params["max_new_tokens"],
    use_cache=hyper_params["use_cache"],
    temperature=hyper_params["temperature"],
    num_beam_groups=hyper_params["num_beam_groups"],
    diversity_penalty=hyper_params["diversity_penalty"],
)
    
# generate responses
for index_i in tqdm(range(dataset_eval.num_rows)):
    input_seq, output_seq = dataset_eval[index_i]['input'], dataset_eval[index_i]['output']
    
    # transform the input sequence if needed
    if input_seq.endswith("title and description."):
        input_seq = "".join(dataset_eval[index_i]['input'].split("\n")[:-1]) + "Please infer the user's preference based on historical purchases and reviews along with the user's intention, and then recommend an item for this user. Please just give the title of the recommended item."


    inputs = tokenizer(
    [
        # "### Input:\n{inputs}\n\n### Response:{outputs}".format(inputs= input, outputs= ""),
        # "{inputs}".format(inputs= input),
        prompt_template.format(input_seq, ""),    
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, generation_config=custom_generation_config)
    
    # init a list to store the ranked generated sequences
    ranked_sequences = []
    for i in range(len(outputs['sequences'])):
        # print(tokenizer.batch_decode(outputs['sequences'][i], skip_special_tokens=True))
        sequence = "".join(tokenizer.batch_decode(outputs['sequences'][i], skip_special_tokens=True))
        sequence = sequence.split("model\n")[1]        
        
        if hyper_params["num_beams_parameter"] == 1: score = outputs['scores'][-1][0][1] # this is for num_beam = 1
        else: score = outputs['sequences_scores'][i]
        
        # transform score from tensor to float
        score = score.item()
        
        # add the generated sequence and its score to the list
        ranked_sequences.append({
            "sequence": sequence,
            "score": score,
        })
        
    # add the generated response to the results dict
    results[index_i] = {
        "raw_input": dataset_eval[index_i]['input'],
        "input": input_seq,
        "ground_truth": output_seq,
        "generated_responses": ranked_sequences,
    }
    
    # check CUDA memory usage
    used_memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 3)
    # if used_memory close to 14GB, empty the cache
    if used_memory > 13.4:
        torch.cuda.empty_cache()
        print("CUDA memory usage is high. Cleared the cache.")

# get current time
now = datetime.now()

# save the results to a json file
save_folder_name = "eval_" + now.strftime("%m%d%y_%H%M%S")
save_folder_path = "eval_results/" + save_folder_name

# make the directory if save_folder_path does not exist
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    
with open(save_folder_path + "/predictions.json", "w") as file:
    json.dump(results, file, indent=4)
    
# save the hyper_params to a json file at the same location
with open(save_folder_path + "/hyper_params.json", "w") as file:
    json.dump(hyper_params, file, indent=4)
