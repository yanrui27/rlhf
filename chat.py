import gradio as gr
import mdtex2html
import requests
import json
import random
import os

BOT_URL = 'http://localhost:8102/generate'

FILE_DATA_PATH = r"datas/竞技场数据.json"
LIMIT_NUM = 1  # 每条数据最多几个人标注
FLAG2IDX = {"A明显好":4, "A更好":3, "A稍微好":2, "俩者都好":1, "都不好":-1, "B稍微好":-2, "B更好":-3, "B明显好":-4, None:None}
IDX2FLAG = dict(zip(FLAG2IDX.values(),FLAG2IDX.keys()))
LEADERBOARD = [[[0]*5 for _ in range(5)] for _ in range(5)]

FLAG2SCORE = {4:[1, 0], 3:[0.8, 0.2], 2:[0.6, 0.4], 1:[0.5, 0.5], -1:[0.5,0.5], -2:[0.4, 0.6], -3:[0.2, 0.8], -4:[0, 1]}

para = [[0.01,0.2,0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.8, 1.0],
        [0.01,0.1,0.3, 0.5, 0.8],
        ]


R_X = {
    'temperature': {k:i for i,k in enumerate(para[0])}, 
    'top_p': {k:i for i,k in enumerate(para[1])},
    'presence_penalty': {k:i for i,k in enumerate(para[2])},
    }


class RandomRemovalList:
    def __init__(self, init_list):
        self.list = init_list
        self.prompt_set = set(range(len(self.list)))
        self.max_prompt_number = len(self.list)
        self.promptID_user_mapping = {x:set() for x in range(self.max_prompt_number )}
        self.user_promptID_mapping = {}
    
    def get_unique(self, user_id):

        if user_id not in self.user_promptID_mapping:
            self.user_promptID_mapping[user_id] = set() 
        
        cur_set = self.prompt_set - self.user_promptID_mapping[user_id]
        if len(cur_set) == 0:
            return -100, None
        
        while True:
            idx = random.sample(cur_set, 1)[0]
            if user_id not in self.promptID_user_mapping[idx]:
                return self.list[idx], idx
            
    def update(self, user_id, idx):
        self.promptID_user_mapping[idx].add(user_id)
        if len(self.promptID_user_mapping[idx]) >= LIMIT_NUM and idx in self.prompt_set:
            self.prompt_set.remove(idx)
        self.user_promptID_mapping[user_id].add(idx)

    def get_id(self, x):
        return self.list.index(x)
            
    
data_list = [json.loads(line) for line in open(FILE_DATA_PATH, "r", encoding="utf-8")]

# flag = 0
# for x in data_list:
#     x["id"] = flag
#     x["response1_para"] = "tmp=0.1"
#     x["response2_para"] = "tmp=0.2"
#     flag+=1  
    
dataset = RandomRemovalList(data_list)


def reset_chatbot(user_id=-1, idx=None, tips1=None, tips2=None, flag=None, savelog=True):
    flag = FLAG2IDX[flag]
    if flag!=None and idx!=-1 and user_id!="":
        if savelog:
            with open("logs/标注记录.log","a",encoding="utf8") as f:
                print(user_id, flag, json.dumps(dataset.list[int(idx)],ensure_ascii=False),file=f)
        dataset.update(user_id, int(idx))
        A_idx = [R_X[k][v] for k, v in json.loads(tips1).items()]
        B_idx = [R_X[k][v] for k, v in json.loads(tips2).items()]
        xa, ya, za = A_idx
        xb, yb, zb = B_idx
        S_A, S_B = FLAG2SCORE[flag]
        E_A = 1 / (1 + 10**((LEADERBOARD[xb][yb][zb] - LEADERBOARD[xa][ya][za]) / 400))
        E_B = 1 / (1 + 10**((LEADERBOARD[xa][ya][za] - LEADERBOARD[xb][yb][zb]) / 400))
        LEADERBOARD[xa][ya][za] = LEADERBOARD[xa][ya][za] + 10 * (S_A - E_A)
        LEADERBOARD[xb][yb][zb] = LEADERBOARD[xb][yb][zb] + 10 * (S_B - E_B)

    data, idx = dataset.get_unique(user_id)
    if type(data) == int:
        if data == -100:
            return [("数据已标完！","...")], [("数据已标完！！","...")], "", "", -1
    
    else:
        if user_id==-1 or user_id=="":
            return [("请先填写用户名！","...")], [("请先填写用户名！！","...")], "", "", -1
        else:
            tips1, tips2 = data["response1_para"], data["response2_para"]
            chatbot1 = [(data["prompt"], data["response1"])]
            chatbot2 = [(data["prompt"], data["response2"])] 
            return chatbot1, chatbot2, tips1, tips2, idx


def show_progress(user_id):
    """ 根据 score_person_id 获取进度 """
    if user_id in dataset.user_promptID_mapping:
        return  f'''- 用户 {user_id} 的标注进度： {len(dataset.user_promptID_mapping[user_id])}/{len(dataset.prompt_set)}'''
                # - 全体用户的标注进度：{len(dataset.list)-len(dataset.prompt_set)}/{len(dataset.prompt_set)}'''
    else:
        return "没有该用户的标注记录"

def show_statistics():
    flat_list = [(value, (i, j, k)) for i, layer in enumerate(LEADERBOARD) for j, row in enumerate(layer) for k, value in enumerate(row)]
    top_20 = sorted(flat_list, key=lambda x: x[0], reverse=True)[:20]

    top_20_indices_pre = ["| Temperature | Top P | Presence Penalty | Score |  \n|-------------|-------|------------------|-------|"]
    top_20_indices = [f"|{para[0][index[0]]} |{para[1][index[1]]} |{para[2][index[2]]} |{value}|" for value, index in top_20]
    return " \n".join(top_20_indices_pre + top_20_indices)



with gr.Blocks() as app:
    gr.HTML("""<h1 align="center">YaYi竞技场</h1>""")


    with gr.Tab("✒️ 数据标注"):
        with gr.Row():
            gr.Markdown("""
                # 雅意模型效果评估平台
                - **明显好**：模型A的回答明显优于模型B的回答，无论是在**内容准确性**、内容完整性、格式规范性或者其他方面，都表现得更好。（注：如果模型B出现非法情况，如重复生成，没有讲完就被截断等，而模型A正常回答问题，那么模型A也是远**明显好**于模型B。）
                - **更好**：模型A的回答在**内容完整性**，格式规范性，都优于模型B的回答，但差距并不明显。
                - **稍微好**：模型A的回答稍微优于模型B的回答，例如**格式规范性**。
                - **俩者都好**：无法确定哪个模型的回答更好，但两者的回复都正常的完成对问题的响应。
                - **都不好**：两个回复内容都存在**重复问题**，**回答偏离问题的主题**，或是生成一些**奇怪的文本**，或者是**没说完**等。
            """)

            with gr.Column(scale=1):
                score_person_id = gr.Text(label="🔑 用户名（输入结束之后，请按一下Enter，会自动刷新数据。）", placeholder="标注者的唯一ID", interactive=True, type="text")
                data_id = gr.Number(label="数据id", visible=True, interactive=False)



        with gr.Row():
            tips1 = gr.Text(label="💡 模型回复A参数", value="", interactive=False, scale=3)
            tips2 = gr.Text(label="💡 模型回复B参数", value="", interactive=False, scale=3)
        
        with gr.Row():
            chatbot1 = gr.Chatbot(label="模型回复A",
                        bubble_full_width=False,
                        rtl=False,
                        avatar_images=(
                        (os.path.join(os.path.dirname(__file__), "human.jpg")), 
                        (os.path.join(os.path.dirname(__file__), "yayi.jpg"))),
                        scale=3,
                        show_copy_button=True)

            chatbot2 = gr.Chatbot(label="模型回复B",
                        bubble_full_width=False,
                        rtl=False,
                        avatar_images=(
                        (os.path.join(os.path.dirname(__file__), "human.jpg")), 
                        (os.path.join(os.path.dirname(__file__), "yayi.jpg"))),
                        scale=3,
                        show_copy_button=True)
        
        with gr.Row():
            submitBtnA4 = gr.Button("A明显好", variant="primary")
            submitBtnA3 = gr.Button("A更好", variant="primary")
            submitBtnA2 = gr.Button("A稍微好", variant="primary")
            submitBtnAB = gr.Button("俩者都好")
            submitBtnB1 = gr.Button("B稍微好", variant="primary")
            submitBtnB2 = gr.Button("B更好", variant="primary")
            submitBtnB3 = gr.Button("B明显好", variant="primary")
            
        with gr.Row():
            submitBtn_no = gr.Button("都不好", variant="primary")
    chatbot1.value, chatbot2.value, tips1.value, tips2.value, data_id.value  = reset_chatbot() 

    submitBtnA4.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnA4] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtnA3.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnA3] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtnA2.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnA2] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtnAB.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnAB] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtnB1.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnB1] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtnB2.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnB2] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtnB3.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtnB3] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    submitBtn_no.click(fn=reset_chatbot, inputs=[score_person_id, data_id, tips1, tips2, submitBtn_no] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
    
    score_person_id.submit(fn=reset_chatbot, inputs=[score_person_id] , outputs=[chatbot1, chatbot2, tips1, tips2, data_id])
 
 
    with gr.Tab("👀 进度查询"):
        with gr.Column():
            score_person_id = gr.Text(label="🔑 用户名", placeholder="标注者的唯一ID", interactive=True, type="text")
            show_progress_button = gr.Button("✅ 点击查询")
            
        with gr.Column():
            progress = gr.Markdown(label="⏩ 当前进度") #, lines=5, placeholder="输入秘钥，点击查询即可返回进度。", interactive=False)
        
        show_progress_button.click(
            fn=show_progress,
            inputs=[score_person_id],
            outputs=[progress]
        )
    
    with gr.Tab("📊 大模型排行榜"):
        with gr.Column():
            show_statistics_button = gr.Button("✅ 点击统计")
            
        with gr.Column():
            statistics = gr.Markdown(label="⏩ 统计结果") # , lines=5, placeholder="输入秘钥，点击查询即可查询全局统计信息。", interactive=False)
        
        show_statistics_button.click(
            fn=show_statistics,
            outputs=[statistics]
        )

    # 初始化，从log中加载数据
    log_use = set()
    for line in open("logs/标注记录.log"):
        if len(line.strip())==0:
            break
        log_data = line.split(" ",maxsplit=2)
        # [score_person_id, data_id, tips1, tips2, submitBtn_no]
        if log_data[0] not in log_use:
            reset_chatbot(log_data[0])
        json_data = json.loads(log_data[2])
        json_data_id = dataset.get_id(json_data)
        reset_chatbot(*[log_data[0], json_data_id, json_data["response1_para"], json_data["response2_para"], IDX2FLAG[int(log_data[1])], False]) 
    chatbot1.value, chatbot2.value, tips1.value, tips2.value, data_id.value = reset_chatbot() 

    
    
app.launch(share=False, server_name="0.0.0.0", show_api=False) #, server_port=10001)