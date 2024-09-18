import json
import csv
import pandas as pd
import copy
import re
from collections import Counter
import math

# n指找到前n个余弦相似度最大的片段，cap是限制用于计算相似度的词频上限
n = 30
cap = 4

pattern1 = r'截至'
pattern2 = r'\d{1,4}年\d{1,2}月\d{1,2}日'

q_file_dir = '/app/intermediate/A02_question_classify_entity.csv'
q_file = pd.read_csv(q_file_dir,delimiter = ",",header = 0)

pdf_csv_file_dir = '/app/data/pdf_analysised'

# 导入模型
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

model_dir = '/root/autodl-tmp/TongyiFinance/Tongyi-Finance-14B-Chat'
# 引入模型提供的分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


# 对c1和c2文本进行余弦相似度的计算
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    
    if magA * magB != 0:
        return dotprod / (magA * magB)
    else:
        return 0


# 创建一个新的csv文件，用于保存余弦相似度前n个最高的文本信息
g = open('/app/intermediate/AB01_question_with_related_text_rp.csv', 'w', newline='', encoding = 'utf-8-sig') 
csvwriter = csv.writer(g)
csvwriter.writerow(['问题id','问题','对应实体','csv文件名','top_n_pages_index','top_n_pages_similarity','top_n_pages'])

# 定义停用词列表，在后续文本中去掉停用词信息
stopword_list = ['根据','招股意见书','招股意向书','报告期内','截至','千元','万元','哪里','哪些','哪个','分别','知道',"什么",'是否','分别','多少','为','?','是','和',
'的','我','想','元','。','？','，','怎样','谁','以及','了','在','哪','对']
bd_list = [',','.','?','。','，','[',']']

print('C02_Started')

# 对于1000个问题通过循环来进行处理
for cyc in range(1000):
    # temp_q为A02文件中当前处理的问题，temp_e为对应实体（公司名）
    temp_q = q_file[cyc:cyc+1]['问题'][cyc]

    # 使用正则表达式移除掉temp_q中的“截至”和日期信息
    str1_list = re.findall(pattern1,temp_q)
    str2_list = re.findall(pattern2,temp_q)
        
    temp_e = q_file[cyc:cyc+1]['对应实体'][cyc]

    # 如果temp_e为N_A，说明该问题没有对应实体，则我们不进行处理
    if temp_e == 'N_A':
        csvwriter.writerow([q_file[cyc:cyc+1]['问题id'][cyc],
                            q_file[cyc:cyc+1]['问题'][cyc],
                            'N_A','N_A','N_A','N_A'])
        continue
    else:
        # 如果temp_e不为N_A，则我们获取该问题对应的csv文件
        temp_csv_dir = pdf_csv_file_dir +'/' + q_file[cyc:cyc+1]['csv文件名'][cyc]
        company_csv = pd.read_csv(temp_csv_dir,delimiter = ",",header = 0)

        # 去除掉temp_q中的空格信息
        temp_q = temp_q.replace(' ','')

        # 去除截至与日期，使得匹配更有针对性
        for word in str1_list:
            temp_q = temp_q.replace(word,'')
        for word in str2_list:
            temp_q = temp_q.replace(word,'')

        # 去除掉temp_q中的temp_e，防止其造成干扰
        temp_q = temp_q.replace(temp_e,' ')
        # 去除掉temp_q中的停用词
        for word in stopword_list:
            temp_q = temp_q.replace(word,' ')

        # 将temp_q按空格分割成单词列表temp_q_list
        temp_q_list = temp_q.split()
        # 定义temp_q_tokens列表，用于保存分词后的token
        temp_q_tokens = list()
        # 遍历问题单词列表temp_q_list，使用分词器对每个单词进行分词和编码，转化为token
        for word in temp_q_list:
            temp_q_tokens_add = tokenizer(word)
            temp_q_tokens_add = temp_q_tokens_add['input_ids']
            for word_add in temp_q_tokens_add:
                temp_q_tokens.append(word_add)

        # 计算每个token出现次数
        C_temp_q_tokens = Counter(temp_q_tokens)

        # list_sim用于存储余弦相似度
        list_sim = list()
        for cyc2 in range(len(company_csv)):
            temp_sim = 0
            temp_file_piece = ''
            if company_csv[cyc2:cyc2+1]['纯文本'][cyc2] == company_csv[cyc2:cyc2+1]['纯文本'][cyc2]:
                temp_file_piece = company_csv[cyc2:cyc2+1]['纯文本'][cyc2]
            if company_csv[cyc2:cyc2+1]['表格'][cyc2] == company_csv[cyc2:cyc2+1]['表格'][cyc2]:
                temp_file_piece = temp_file_piece + company_csv[cyc2:cyc2+1]['表格'][cyc2].replace('None'," ")
            
            for bd in bd_list:
                temp_file_piece = temp_file_piece.replace(bd,' ')
                
            temp_s_tokens = tokenizer(temp_file_piece)
            temp_s_tokens = temp_s_tokens['input_ids']
            
            C_temp_s_tokens = Counter(temp_s_tokens)
            C_temp_s_tokens['220'] = 0

            # 如果词频数量大于cap，将其设为cap，防止词频次数过大影响结果
            for token in C_temp_s_tokens:
                if C_temp_s_tokens[token] >= cap:
                    C_temp_s_tokens[token] = cap

            # 如果token列表为空，则余弦相似度为0，否则，计算余弦相似度
            if temp_q_tokens == '':
                temp_sim = 0
            else:
                temp_sim = counter_cosine_similarity(C_temp_q_tokens,C_temp_s_tokens)
            list_sim.append(temp_sim)
            
        # 找到相似度最大的的前n个文本
        t = copy.deepcopy(list_sim)
        # max_number存储余弦相似度，max_index存储对应索引
        max_number = []
        max_index = []
        
        for _ in range(n):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        t = []
        
        # temp_file_pieces_list存储要保存到文件的各项信息
        temp_file_pieces_list = list()
        for index in max_index:
            temp_dict = {}
            if company_csv[index:index+1]['纯文本'][index] == company_csv[index:index+1]['纯文本'][index]:
                temp_dict['text'] = company_csv[index:index+1]['纯文本'][index]
            if company_csv[index:index+1]['表格'][index] == company_csv[index:index+1]['表格'][index]:
                temp_dict['table'] = company_csv[index:index+1]['表格'][index].replace('None'," ")
            
            temp_file_pieces_list.append(temp_dict)

        # 将对应的片段放入文件
        csvwriter.writerow([q_file[cyc:cyc+1]['问题id'][cyc],
                    q_file[cyc:cyc+1]['问题'][cyc],
                    temp_e,q_file[cyc:cyc+1]['csv文件名'][cyc],max_index,max_number,temp_file_pieces_list])
g.close()  
                      