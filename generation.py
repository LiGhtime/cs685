import json

# asins_small.json to be used for meta data matching
json_file_asins = './data/asins_small.json'
with open(json_file_asins, "r") as file:
    asin_dict = json.load(file)

# read eval dataset from local
from datasets import load_from_disk

dataset_eval = load_from_disk("./data/gemma_chat_eval")
print("Loaded eval dataset from {}".format("./data/gemma_chat_eval"))

index_i = 16
dataset_eval[index_i]['input'], \
dataset_eval[index_i]['output']

if True:
    from transformers import TextStreamer, GenerationConfig
    from unsloth import FastLanguageModel
    
    max_seq_length = 4096 # 8192 | Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "outputs/model_04242024_090830/", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
prompt_template = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n{}"

input = dataset_eval[index_i]['input']
# input = dataset_eval[index_i]['input'] + "\n Please also explain yourself in one sentence or two, why the user might be interested in your recommended item."
# input = "Hello, I am looking for a movie that will make me feel happy and excited. I love action movies with a lot of suspense and thrill. I also enjoy movies with a lot of drama and romance. Can you recommend a movie that will make me feel happy and excited? And please explain why you recommend this movie."
# input= "Below is the previous historical purchases and reviews of the user:\n```\nItem title: One Step Behind \n Item description: A dangerous man awakes in the care of a mysterious woman. Once the pieces of his past fall into place, he's faced with the stark choice of accepting his new found love or becoming the man he once was. \n rating: 4.0 \n review: Nice Hard Boiled Indie! Plenty of Bad Guys, Femme Fatales and one Big Bad Ass Reluctant Hero! If you like old school Film Noir -  more specifically Hard Boiled - you'll enjoy this little indie film. It reminded me a lot of the Coen Brother's BLOOD SIMPLE and another desert noir from the 90s starring Nick Cage RED ROCK WEST. It has that same pace and flavor to it and it pays of at the end. Well done!-------\nItem title: Untouched \n Item description: This legal drama explores the life-changing effects a secret abortion has on Mitch Thomas. A town startled by an infant found murdered in a dumpster, stirs Mitch, the reluctant attorney, conflicted by the love he lost, to unveil his haunting secret. \n rating: 4.0 \n review: Nice courtroom drama with it's heart in the right place. Beautifully shot and written, this is an intriguing story with a bittersweet ending. Really enjoyed it!-------\nItem title: The Man In The Silo \n Item description: A successful business man wakes up to find himself in a dilapidated grain silo, he must reconnect the dots of his traumatized memory to discover the truth of how he ended up there. \n rating: 5.0 \n review: This is a beautiful film full of great cinematic storytelling plus a classic re-hash of the Bernard Herman's Vertigo Score. Being a Hitchcock fan, I can honestly say that this is a true HITCHCOCKIAN film - which we hear a lot but hardly ever see. You can tell the film was made with so much attention to detail and passion by the great cinematography, transitions, acting, pace, acting and editing - and everything works together to achieve a great visual art piece. But don't get me wrong, it is also super exciting and entertaining as the film keeps you guessing from beginning to end. It is full of intrigue and suspense - and it is haunting. I highly recommend THE MAN IN THE SILO and I can almost guarantee you haven't seen anything like it!-------\n```\nAnd here is the user's intention: I really enjoy indie films with strong storytelling and emotional depth, so I'm looking for a movie that will steal my heart and make me want to fall in love all over again.\nPlease infer the user's preference based on historical purchases and reviews along with the user's intention, and then recommend an item for this user and provide its description."
# input = "hello, can you recommend me a random movie? Maybe a romantic one? And please describe the reason why you recommend this movie."

inputs = tokenizer(
[
    # "### Input:\n{inputs}\n\n### Response:{outputs}".format(inputs= input, outputs= ""),
    # "{inputs}".format(inputs= input),
    prompt_template.format(input, ""),    
], return_tensors = "pt").to("cuda")

num_beams_parameter = 3
custom_generation_config = GenerationConfig(
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    asin_dict=asin_dict,
    tokenizer=tokenizer,
    return_dict_in_generate=True,
    output_scores=True,
    # output_logits=True,
    do_sample=True,
    num_beams=num_beams_parameter, 
    num_return_sequences=num_beams_parameter,
    max_new_tokens=96,
    use_cache=True,
    temperature=1,
    # num_beam_groups=num_beams_parameter,
    # diversity_penalty=0.3,
)

inputs = tokenizer(
[
    prompt_template.format(input, ""),    
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, generation_config=custom_generation_config)
print(outputs)
print(tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True))
# outputs['scores'] is the tuple of probabilty distributions at every time step.

# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048, num_beams=1, do_sample=True, generation_config=custom_generation_config, use_cache=True)