import json
import numpy as np
import datasets
import faiss     
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pickle


def get_prompt(data):
    prompt = ""
    for x in data["conversations"][:-1]:
        prompt += x['from'].capitalize() +":\n" + x["value"] + "\n"
        break
    return prompt

DATA_FILE = "train_3.5M_CN.json"
SAVE_FILE = "k_sft_embeddings.pkl"

if __name__ =="__main__":

    prompt_list = []
    input_json_data = open(DATA_FILE).readlines()
    for line in input_json_data:
        data = json.loads(line)
        prompt = get_prompt(data)
        prompt_list.append(prompt)


    # 获取embeddings
    model = SentenceTransformer('bge-base-zh').cuda()

    # embeddings = model.encode(prompt_list, normalize_embeddings=True,batch_size=1024,show_progress_bar=True)
    # 开启GPU加速
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(prompt_list, pool,batch_size=1024)
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
    


        