import requests
import json
import pandas
import tqdm
import datasets
import re
import tqdm
import random
from datetime import datetime
import uuid
import hashlib
import hmac
import base64
import time
import concurrent.futures

url = f'https://tilake.wenge.com/saas-gateway/503295a559ae022610935e1e52ee10b5/generate'
#################
wx_idx2str = {
    0:"temperature",
    1:"top_p",
    2:"presence_penalty"
}

para = [[0.01,0.2,0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.8, 1.0],
        [0.01,0.1,0.3, 0.5, 0.8],
        ]

def runs(prompt):
    w1_idx, w2_idx = random.sample([0,1,2], 2)
    w3_idx = 3 - w1_idx -w2_idx
    
    w1 = para[w1_idx][random.sample(range(5), 1)[0]]
    w2 = para[w2_idx][random.sample(range(5), 1)[0]]
    
    cur_para = {
        wx_idx2str[w1_idx]:w1,
        wx_idx2str[w2_idx]:w2,
    }
    
    w3_pair = [para[w3_idx][x] for x in random.sample(range(5), 2)]
    
    res = []
    for w3 in w3_pair:
        data = {
        "id":"yanrui",
        "do_sample": True,
        "max_new_tokens": 2048,
        "best_of": 5
        }
        data.update(cur_para)
        data["messages"]= [{"role":"user","content":prompt}]
        data[wx_idx2str[w3_idx]] = w3


        # 构造请求头
        http_method = 'POST'
        api_secret = '43b1272229014b22ba93ab8a7985a522'
        path = '/503295a559ae022610935e1e52ee10b5/generate'

        DATE = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        CONTENT_TYPE = 'application/json'
        ACCEPT = 'application/json'

        # 将请求信息组合成一个字符串
        data_to_sign = f'{http_method}\n{ACCEPT}\n{CONTENT_TYPE}\n{DATE}\n{path}'

        # 使用HMAC-SHA256算法进行签名
        signature = hmac.new(api_secret.encode(), data_to_sign.encode(), hashlib.sha256).digest()
        # 将签名进行Base64编码
        signature_base64 = base64.b64encode(signature).decode()

        # 注意：需要在请求头中包含你在签名中使用的信息
        headers = {
            'x-tilake-app-key': '4d2e810f2059467eb3c6535bfe534b8f', 
            'x-tilake-ca-signature-method': 'HmacSHA256',
            'x-tilake-ca-timestamp': str(int(time.time() * 1000)),
            'x-tilake-ca-nonce': str(uuid.uuid4()),
            'x-tilake-ca-signature': signature_base64,
            'Date': DATE,
            'Content-Type': CONTENT_TYPE,
            'Accept': ACCEPT,
        }

        # 发起请求
        response = requests.post(url, json=data, headers=headers)
        response = json.loads(response.text)["data"]['choices'][0]["message"]["content"]
        res.append([response, {v:data[v] for k,v in wx_idx2str.items()}])


    return {
            "prompt":prompt,
            "response1":res[0][0],
            "response1_para":json.dumps(res[0][1], ensure_ascii=False),
            "response2":res[1][0],
            "response2_para":json.dumps(res[1][1], ensure_ascii=False),
            }

if __name__ == "__main__":
    prompt_list = open("1k_query.json",encoding="utf8").readlines()
    prompt_list = list(map(json.loads, prompt_list))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm.tqdm(executor.map(runs, prompt_list), total=len(prompt_list)))