import csv
import json
import pandas as pd
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

csv_file_1_dir = "/root/llm/processed_data/FA_V5_SQL.csv"
csv_file_1 = pd.read_csv(csv_file_1_dir, delimiter=",", header=0)
csv_file_2_dir = "/root/llm/processed_data/FA_V5_Text_cap4_4_nt.csv"
csv_file_2 = pd.read_csv(csv_file_2_dir, delimiter=",", header=0)

model_dir = '/root/autodl-tmp/TongyiFinance/Tongyi-Finance-14B-Chat'
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir,
                                                           trust_remote_code=True,
                                                           temperature = 0.0001,
                                                           top_p = 1,
                                                           do_sample = False,
                                                           seed = 1234)

# 函数：使用模型生成综合答案
def generate_summary(answers, question):
    answers_text = '\n'.join(answers)
    prompt = (f"关于问题：{question}\n"
              f"以下是多个可能的答案：\n{answers_text}\n"
              f"请根据这些答案提供一个简明准确的综合答案。")
    temp_FA, history = model.chat(tokenizer, prompt, history=None)
    return temp_FA

# 处理函数：将多个答案综合成一个
def process_answers(temp_answer):
    answers = temp_answer.split('\n')
    if len(answers) > 1:
        temp_answer = generate_summary(answers)
    return temp_answer


print('D01_started')
list_of_items = list()
for cyc in range(1000):
    temp_dict = {}
    temp_dict['id'] = str(csv_file_1['问题id'][cyc])
    temp_dict["question"] = csv_file_1['问题'][cyc]
    temp_answer = ""
    if csv_file_2[cyc:cyc + 1]['实体答案'][cyc] != 'N_A':
        temp_answer = str(csv_file_2[cyc:cyc + 1]['final_ans1'][cyc])
        temp_answer = process_answers(temp_answer)  # 处理多个答案

    else:
        temp_answer = str(csv_file_1[cyc:cyc + 1]['FA'][cyc])

    for cyc in range(10):
        temp_answer = temp_answer.replace("根据资料%s" % str(cyc), '')

    temp_dict["answer"] = temp_answer
    list_of_items.append(temp_dict)

with open('/root/llm/result1.jsonl', mode='w', encoding='utf-8') as f:
    for cyc in range(1000):
        temp_dict = list_of_items[cyc]
        temp_str = json.dumps(temp_dict, ensure_ascii=False).replace('{"id": "', '{"id": ').replace('", "question":',
                                                                                                    ', "question":')
        f.write(temp_str + '\n')

f.close()
exit()
