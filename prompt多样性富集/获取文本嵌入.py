import json
import numpy as np
import datasets
import faiss     
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pickle
import tqdm

def get_prompt(data):
    prompt = ""
    for x in data["conversations"][:-1]:
        prompt += x['from'].capitalize() +":\n" + x["value"] + "\n"
        break
    return prompt

DATA_FILE = "train_3.5M_CN.json"
# DATA_FILE = "100k.json"
SAVE_FILE = "k_sft_embeddings.pkl"

if __name__ =="__main__":

    prompt_list = []
    input_json_data = open(DATA_FILE).readlines()
    prompt_list = [get_prompt(json.loads(line)) for line in input_json_data]
    

    # 获取embedding_model
    model = SentenceTransformer('bge-base-zh').cuda()

    # embeddings = model.encode(prompt_list, normalize_embeddings=True,batch_size=1024,show_progress_bar=True)
    # 开启GPU加速
    pool = model.start_multi_process_pool()

    # Define the batch size
    batch_size = 1024 * len(pool['processes']) * 4


    # Calculate the number of batches
    num_batches = len(prompt_list) // batch_size + (len(prompt_list) % batch_size != 0)

    # Create an empty list to store the embeddings
    embeddings = []
    # Process each batch
    for i in tqdm.tqdm(range(num_batches)):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(prompt_list))
        batch = prompt_list[start_index:end_index]
        batch_embeddings = model.encode_multi_process(batch, pool, batch_size=1024)
        embeddings.append(batch_embeddings)

    embeddings = np.concatenate(embeddings)
    import gc
    import torch
    del model 
    torch.cuda.empty_cache()
    gc.collect() 
    torch.cuda.empty_cache()
    

    pickle.dump({
                "embeddings": embeddings,
                "prompt_list": prompt_list,
                "input_json_data":input_json_data,
            },
            open(f"{len(embeddings)//1000}"+SAVE_FILE,"wb"))
    


        