import pickle
import faiss
import numpy as np
import json
import logging

logging.basicConfig(level=logging.DEBUG,  # 设置日志级别为 DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILE = "100k_sft_embeddings.pkl"
SAVE_FILE = "yanrui_去重结果.json"


if __name__ == "__main__":
    kkk = 0
    embeddings, prompt_list, input_json_data = None, None, None
    checkpoint = pickle.load(open(DATA_FILE,"rb"))
    for key in checkpoint:
        locals()[key] = checkpoint[key]
    data_ntotal = len(embeddings)
    #embeddings = embeddings[:1000,] # 减少数据量，用于加速调试过程


    logging.info(f"开始建立faiss-IVF索引！")
    quantizer = faiss.IndexFlatL2(768)  
    index = faiss.IndexIVFFlat(quantizer, 768, data_ntotal//100, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)               
    index.nprobe = 5   
    
    while 1:
        index.reset()
        index.add(embeddings) 
        pickle.dump({
                "embeddings": embeddings,
                "prompt_list": prompt_list,
                "input_json_data":input_json_data,
            },
            open(f"checkpoint.pkl","wb"))
        logging.info(f"Round{kkk} 开始查找相似向量！")
        del_list = []           
        D, I = index.search(embeddings, 2)
        cur_embedding_ntotal = len(embeddings)
        for (i, j), s in zip(I[:,:2],D[:,1]):
            if j > i and s >0.9:
                del_list.append(i)

        if len(del_list) == 0:
            break
        
        logging.info(f"Round{kkk} 开始去重！")
        mask = np.ones(len(embeddings), dtype=bool)
        mask[del_list] = False
        embeddings = embeddings[mask]
        prompt_list = [prompt_list[i] for i in range(len(prompt_list)) if mask[i]]
        input_json_data = [input_json_data[i] for i in range(len(input_json_data)) if mask[i]]


        logging.info(f"Round{kkk} 成功过滤{len(del_list)}个prompt！\n")
        kkk +=1 

    
    logging.info(f"存储去重结果到文件:\t{SAVE_FILE}\n\n最终存储数据个数:\t{len(input_json_data)}(/{data_ntotal})")    
    open(SAVE_FILE,"w").wirte("".join(input_json_data)).close()
