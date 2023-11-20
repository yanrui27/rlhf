import pickle
import faiss
import numpy as np
import json

DATA_FILE = "100k_sft_embeddings.pkl"

if __name__ == "__main__":
    kkk = 0
    embeddings, prompt_list, input_json_data = None, None, None
    checkpoint = pickle.load(open(DATA_FILE,"rb"))
    for key in checkpoint:
        locals()[key] = checkpoint[key]


    while 1:
        pickle.dump({
                "embeddings": embeddings,
                "prompt_list": prompt_list,
                "input_json_data":input_json_data,
            },
            open(f"checkpoint.pkl","wb"))
        
        print(f"Round{kkk} 开始建立faiss索引！")
        quantizer = faiss.IndexFlatL2(768)  
        index = faiss.IndexIVFFlat(quantizer, 768, 1000, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)                
        index.nprobe = 10   

        print(f"Round{kkk} 开始查找相似向量！")
        del_list = []           
        D, I = index.search(embeddings, 2)
        for (i, j), s in zip(I[:,:2],D[:,1]):
            #print(prompt_list[i],prompt_list[j],s,"\n\n")
            if j > i and s >0.9:
                del_list.append(i)

        if del_list == []:
            break
        
        print(f"Round{kkk} 开始去重！")
        for idx in sorted(del_list, reverse=True):
            embeddings = np.delete(embeddings, idx, axis=0)
            del prompt_list[idx]
            del input_json_data[idx]

        print(f"Round{kkk} 成功过滤{len(del_list)}个prompt！\n")
        kkk +=1 

        
    json.dump(open("yanrui_去重结果.json","w") , "".join(input_json_data), ensure_ascii=False)
