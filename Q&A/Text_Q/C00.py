import os
import csv
import json
from collections import Counter
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

# 定义文本文件夹路径和输出 CSV 文件路径
text_folder = '/path/to/txt2csv_normalized'  # 替换为实际的路径
output_csv = '/path/to/AD_normalized_ot.csv'  # 替换为实际的路径

# 加载 tokenizer
model_dir = '/root/autodl-tmp/TongyiFinance/Tongyi-Finance-14B-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 加载所有文本文件并计算全局词频
global_counter = Counter()

documents = []
file_names = []
for text_file in os.listdir(text_folder):
    if text_file.endswith('.csv'):
        file_path = os.path.join(text_folder, text_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            text = " ".join(row['纯文本'] for row in reader if '纯文本' in row)
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            global_counter.update(token_ids)

            documents.append((text, token_ids))
            file_names.append(os.path.basename(text_file))

# 准备CSV文件的写入
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['文件名', 'normalized'])  # 写入表头

    # 遍历每个文档，计算标准化标记频率
    for (text, token_ids), file_name in zip(documents, file_names):
        # 计算标记频率
        counter = Counter(token_ids)

        # 使用改进公式计算标准化标记频率
        total_count = sum(counter.values())
        normalized_counter = {}
        for token, count in counter.items():
            fa_t = global_counter[token]
            if fa_t != 0:
                normalized_counter[token] = count / total_count * (fa_t ** -2)
            else:
                normalized_counter[token] = count / total_count

        # 将 Counter 对象转换为 JSON 字符串，以便于存储在 CSV 文件中
        normalized_str = json.dumps(normalized_counter)

        # 将文件名和标准化标记频率添加到数据列表
        csvwriter.writerow([file_name, normalized_str])

print(f"CSV 文件 '{output_csv}' 已创建。")
