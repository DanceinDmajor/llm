import json
import csv
import pandas as pd
import copy
import re
from collections import Counter
import math
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from share import tokenizer
from share import model


stopword_list = ['根据', '招股意见书', '招股意向书', '报告期内', '截至', '千元', '万元', '哪里', '哪些', '哪个', '分别',
                 '知道', "什么", '是否', '分别', '多少', '为', '?', '是', '和',
                 '的', '我', '想', '元', '。', '？', '，', '怎样', '谁', '以及', '了', '在', '哪', '对']
bd_list = [',', '.', '?', '。', '，', '[', ']']
unknown_words = ["不知道", "无法直接", "无法确定", "没有找到", '未知', '与问题无关', '无法直接得出', '没有具体说明',
                 '没有看到', '没有被提及', '没有发现', '不清楚', '没有明确', '无法得出', '尚未公布', '●●●',
                 '不在提供的资料',
                 '无法回答', '可以推算', '无法找到', '没有提供', '无法直接', '未知的', '未在资料中', '没有在资料中',
                 '未明确列出', '没有被明确', '没有被提供', '无法计算', '我找不到', '无法浏览网页', ']]', '[[',
                 '并未给出',
                 '无法判断', '没有直接给出', '没有提到', '我不清楚', '无法准确回答', '没有直接说明', '未被提及',
                 '无法访问', '无法给出', '无法得出', '无法预测', '无法提供']


def counter_cosine_similarity(c1, c2, normalized_dict=None):
    terms = set(c1).union(c2)
    if normalized_dict:
        dotprod = sum(c1.get(k, 0) * c2.get(k, 0) / normalized_dict.get(k, 1) for k in terms)
        magA = math.sqrt(sum(c1.get(k, 0) ** 2 / (normalized_dict.get(k, 1) ** 2) for k in terms))
        magB = math.sqrt(sum(c2.get(k, 0) ** 2 / (normalized_dict.get(k, 1) ** 2) for k in terms))
    else:
        dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
        magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
        magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))

    if magA * magB != 0:
        return dotprod / (magA * magB)
    else:
        return 0


def process_text_question(question):
    q_file_dir = '/root/llm/processed_data/A02_question_classify_entity.csv'
    q_file = pd.read_csv(q_file_dir, delimiter=",", header=0)
    normalized_dir = '/root/llm/data/AD_normalized_ot.csv'
    normalized_file = pd.read_csv(normalized_dir, delimiter=",", header=0)
    n_list_1 = list(normalized_file['文件名'])
    n_list_2 = list(normalized_file['normalized'])
    pdf_csv_file_dir = '/root/llm/data/txt2csv_normalized'

    for cyc in range(1000):
        temp_q = q_file.iloc[cyc]['问题']
        temp_e = q_file.iloc[cyc]['对应实体']

        if temp_e == 'N_A':
            continue

        temp_csv_dir = pdf_csv_file_dir + '/' + q_file.iloc[cyc]['csv文件名']
        company_csv = pd.read_csv(temp_csv_dir, delimiter=",", header=0)
        temp_hash = q_file.iloc[cyc]['csv文件名'][:-8] + '.txt'
        normalized_id = n_list_1.index(temp_hash)
        normalized_dict = eval(n_list_2[normalized_id])
        temp_q = temp_q.replace(' ', '')

        for word in stopword_list:
            temp_q = temp_q.replace(word, ' ')
        temp_q_list = temp_q.split()
        temp_q_tokens = [token for word in temp_q_list for token in tokenizer(word)['input_ids']]
        C_temp_q_tokens = Counter(temp_q_tokens)

        list_sim = []
        for _, row in company_csv.iterrows():
            temp_file_piece = row['纯文本']
            for bd in bd_list:
                temp_file_piece = temp_file_piece.replace(bd, ' ')
            temp_s_tokens = tokenizer(temp_file_piece)['input_ids']
            C_temp_s_tokens = Counter(temp_s_tokens)
            temp_sim = counter_cosine_similarity(C_temp_q_tokens, C_temp_s_tokens, normalized_dict)
            list_sim.append(temp_sim)

        max_index = sorted(range(len(list_sim)), key=lambda i: list_sim[i], reverse=True)[:30]
        temp_file_pieces_list = [{'text': company_csv.iloc[i]['纯文本']} for i in max_index]

        response_list = generate_answers(temp_file_pieces_list, temp_q, 30, 10)
        answer = "\n".join(response_list)

        return answer


def generate_answers(piece_list, question, n, m):
    temp_index = 0
    return_response_list = []

    while temp_index < n:
        next_piece = ''
        prompt = ("你是一个人工智能文字秘书。下面是一段资料，不要计算，不要计算，直接从资料中寻找问题的答案，"
                  "使用完整的句子回答问题。\n 如果资料不包含问题的答案，回答“不知道。”如果从资料无法得出问题的答案，"
                  "回答“不知道。”如果答案未在资料中说明，回答“不知道。”如果资料与问题无关或者在资料中找不到问题的答案，"
                  "回答“不知道。”如果资料没有明确说明问题答案，回答“不知道。”资料：")
        if 'text' in piece_list[temp_index]:
            next_piece += piece_list[temp_index]['text']

        prompt += f'资料：{next_piece} \n 问题：{question}'
        response, _ = model.chat(tokenizer, prompt, history=None)
        response = response.replace('\n', '').strip()

        if len(response) < 500 and all(word not in response for word in unknown_words):
            if response not in return_response_list:
                return_response_list.append(response)

        if len(return_response_list) >= m:
            break

        temp_index += 1

    return return_response_list


def find_relevant_pieces(question, entity, csv_filename):
    # 定义内部变量
    pattern1 = r'截至'
    pattern2 = r'\d{1,4}年\d{1,2}月\d{1,2}日'


    normalized_dir = '/root/llm/data/AD_normalized_ot.csv'
    normalized_file = pd.read_csv(normalized_dir, delimiter=",", header=0)
    n_list_1 = list(normalized_file['文件名'])
    n_list_2 = list(normalized_file['normalized'])

    pdf_csv_file_dir = '/root/llm/data/txt2csv_normalized'

    stopword_list = ['根据', '招股意见书', '招股意向书', '报告期内', '截至', '千元', '万元', '哪里', '哪些', '哪个', '分别', '知道', "什么", '是否', '分别', '多少', '为', '?', '是', '和',
                     '的', '我', '想', '元', '。', '？', '，', '怎样', '谁', '以及', '了', '在', '哪', '对']
    bd_list = [',', '.', '?', '。', '，', '[', ']']

    results = []

    temp_q = question
    temp_e = entity

    if temp_e == 'N_A':
        results.append({
            '问题': question,
            '对应实体': 'N_A',
            'csv文件名': 'N_A',
            'top_n_pages_index': 'N_A',
            'top_n_pages_similarity': 'N_A',
            'top_n_pages': 'N_A'
        })
    else:
        temp_csv_dir = pdf_csv_file_dir + '/' + csv_filename
        company_csv = pd.read_csv(temp_csv_dir, delimiter=",", header=0)
        temp_hash = csv_filename[0:-8] + '.txt'

        normalized_id = n_list_1.index(temp_hash)
        normalized_dict = eval(n_list_2[normalized_id])
        company_csv = pd.read_csv(temp_csv_dir, delimiter=",", header=0)
        temp_q = temp_q.replace(' ', '')

        temp_q = temp_q.replace(temp_e, ' ')
        for word in stopword_list:
            temp_q = temp_q.replace(word, ' ')
        temp_q_list = temp_q.split()
        temp_q_tokens = []

        for word in temp_q_list:
            temp_q_tokens_add = tokenizer(word)
            temp_q_tokens_add = temp_q_tokens_add['input_ids']
            temp_q_tokens.extend(temp_q_tokens_add)

        C_temp_q_tokens = Counter(temp_q_tokens)
        list_sim = []

        for cyc2 in range(len(company_csv)):
            temp_sim = 0
            temp_file_piece = ''
            if company_csv.iloc[cyc2]['纯文本'] == company_csv.iloc[cyc2]['纯文本']:
                temp_file_piece = company_csv.iloc[cyc2]['纯文本']

            for bd in bd_list:
                temp_file_piece = temp_file_piece.replace(bd, ' ')

            temp_s_tokens = tokenizer(temp_file_piece)
            temp_s_tokens = temp_s_tokens['input_ids']

            C_temp_s_tokens = Counter(temp_s_tokens)
            C_temp_s_tokens['220'] = 0

            if not temp_q_tokens:
                temp_sim = 0
            else:
                temp_sim = counter_cosine_similarity(C_temp_q_tokens, C_temp_s_tokens, normalized_dict)
            list_sim.append(temp_sim)

        max_number = []
        max_index = []

        for _ in range(30):
            number = max(list_sim)
            index = list_sim.index(number)
            list_sim[index] = 0
            max_number.append(number)
            max_index.append(index)

        temp_file_pieces_list = []
        for index in max_index:
            temp_dict = {}
            if company_csv.iloc[index]['纯文本'] == company_csv.iloc[index]['纯文本']:
                temp_dict['text'] = company_csv.iloc[index]['纯文本']
            temp_file_pieces_list.append(temp_dict)

        results = {
            '问题': question,
            '对应实体': temp_e,
            'csv文件名': csv_filename,
            'top_n_pages_index': max_index,
            'top_n_pages_similarity': max_number,
            'top_n_pages': temp_file_pieces_list
        }
    return results

pdf_csv_file_dir = '/root/llm/data/pdf_analysised'

def find_relevant_pieces_from_pdf(question, entity, csv_filename):
    n = 30
    cap = 4

    pattern1 = r'截至'
    pattern2 = r'\d{1,4}年\d{1,2}月\d{1,2}日'

    stopword_list = ['根据', '招股意见书', '招股意向书', '报告期内', '截至', '千元', '万元', '哪里', '哪些', '哪个',
                     '分别', '知道', "什么", '是否', '分别', '多少', '为', '?', '是', '和',
                     '的', '我', '想', '元', '。', '？', '，', '怎样', '谁', '以及', '了', '在', '哪', '对']
    bd_list = [',', '.', '?', '。', '，', '[', ']']
    str1_list = re.findall(pattern1, question)
    str2_list = re.findall(pattern2, question)

    if entity == 'N_A':
        return {'question_id': None, 'question': question, 'entity': 'N_A', 'csv_filename': 'N_A',
                'top_n_pages_index': 'N_A', 'top_n_pages_similarity': 'N_A', 'top_n_pages': 'N_A'}

    temp_csv_dir = pdf_csv_file_dir + '/' + csv_filename
    company_csv = pd.read_csv(temp_csv_dir, delimiter=",", header=0)
    question = question.replace(' ', '')
    for word in str1_list:
        question = question.replace(word, '')
    for word in str2_list:
        question = question.replace(word, '')

    question = question.replace(entity, ' ')
    for word in stopword_list:
        question = question.replace(word, ' ')

    question_list = question.split()
    question_tokens = list()
    for word in question_list:
        question_tokens_add = tokenizer(word)
        question_tokens_add = question_tokens_add['input_ids']
        for word_add in question_tokens_add:
            question_tokens.append(word_add)

    C_question_tokens = Counter(question_tokens)
    list_sim = list()
    for i in range(len(company_csv)):
        temp_sim = 0
        temp_file_piece = ''
        if company_csv.iloc[i]['纯文本'] == company_csv.iloc[i]['纯文本']:
            temp_file_piece = company_csv.iloc[i]['纯文本']
        if company_csv.iloc[i]['表格'] == company_csv.iloc[i]['表格']:
            temp_file_piece = temp_file_piece + company_csv.iloc[i]['表格'].replace('None', " ")

        for bd in bd_list:
            temp_file_piece = temp_file_piece.replace(bd, ' ')

        temp_s_tokens = tokenizer(temp_file_piece)
        temp_s_tokens = temp_s_tokens['input_ids']

        C_temp_s_tokens = Counter(temp_s_tokens)
        C_temp_s_tokens['220'] = 0

        for token in C_temp_s_tokens:
            if C_temp_s_tokens[token] >= cap:
                C_temp_s_tokens[token] = cap

        if question_tokens == '':
            temp_sim = 0
        else:
            temp_sim = counter_cosine_similarity(C_question_tokens, C_temp_s_tokens)
        list_sim.append(temp_sim)

    t = copy.deepcopy(list_sim)
    max_number = []
    max_index = []

    for _ in range(n):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)

    temp_file_pieces_list = list()
    for index in max_index:
        temp_dict = {}
        if company_csv.iloc[index]['纯文本'] == company_csv.iloc[index]['纯文本']:
            temp_dict['text'] = company_csv.iloc[index]['纯文本']
        if company_csv.iloc[index]['表格'] == company_csv.iloc[index]['表格']:
            temp_dict['table'] = company_csv.iloc[index]['表格'].replace('None', " ")
        temp_file_pieces_list.append(temp_dict)

    return {
        'question_id': None,
        'question': question,
        'entity': entity,
        'csv_filename': csv_filename,
        'top_n_pages_index': max_index,
        'top_n_pages_similarity': max_number,
        'top_n_pages': temp_file_pieces_list
    }



def answer_generator(piece_list, temp_q, n=20, m=10):
    temp_index = 0
    return_response_list = []
    unknown_words = ["不知道", "无法直接", "无法确定", "没有找到", '未知', '与问题无关', '无法直接得出', '没有具体说明',
                     '没有看到', '没有被提及', '没有发现', '不清楚', '没有明确', '无法得出', '尚未公布', '●●●', '不在提供的资料',
                     '无法回答', '可以推算', '无法找到', '没有提供', '无法直接', '未知的', '未在资料中', '没有在资料中',
                     '未明确列出', '没有被明确', '没有被提供', '无法计算', '我找不到', '无法浏览网页', ']]', '[[', '并未给出',
                     '无法判断', '没有直接给出', '没有提到', '我不清楚', '无法准确回答', '没有直接说明', '未被提及', '无法访问', '无法给出', '无法得出', '无法预测', '无法提供']
    prompt_prefix = "你是一个人工智能文字秘书。下面是一段资料，不要计算，不要计算，直接从资料中寻找问题的答案，使用完整的句子回答问题。\n 如果资料不包含问题的答案，回答“不知道。”如果从资料无法得出问题的答案，回答“不知道。”如果答案未在资料中说明，回答“不知道。”如果资料与问题无关或者在资料中找不到问题的答案，回答“不知道。”如果资料没有明确说明问题答案，回答“不知道。”资料："

    while temp_index < n:
        next_piece = ''
        if 'text' in piece_list[temp_index]:
            next_piece += piece_list[temp_index]['text'] if piece_list[temp_index]['text'] == piece_list[temp_index]['text'] else ''
        if 'table' in piece_list[temp_index]:
            next_piece += piece_list[temp_index]['table'] if piece_list[temp_index]['table'] == piece_list[temp_index]['table'] else ''
        temp_index += 1

        prompt = prompt_prefix + '：' + next_piece + ' \n 问题：' + temp_q
        response, history = model.chat(tokenizer, prompt, history=None)
        response = response.replace('\n', '')

        if all(word not in response for word in unknown_words) and response not in return_response_list and len(response) < 500:
            return_response_list.append(response)
        if len(return_response_list) >= m:
            break

    return return_response_list

# 函数：使用模型生成综合答案
def generate_summary(answers, question):
    answers_text = '\n'.join(answers)
    prompt = (f"关于问题：{question}\n"
              f"以下是多个可能的答案：\n{answers_text}\n"
              f"请根据这些答案提供一个简明准确的综合答案。")
    temp_FA, history = model.chat(tokenizer, prompt, history=None)
    return temp_FA

# 处理函数：将多个答案综合成一个
def process_answers(temp_answer,question):
    answers = temp_answer.split('\n')
    if len(answers) > 1:
        temp_answer = generate_summary(answers,question)
    return temp_answer

def generate_final_answers(question, entity, top_n_pages, top_n_pages_2, n1=20, n2=20, num_of_answer=10):
    answer = 'N_A'
    response_list = []

    if entity != 'N_A':
        response_list = answer_generator(top_n_pages, question, n1, num_of_answer)

        if len(response_list) <= 7:
            response_list_2 = answer_generator(top_n_pages_2, question, n2, num_of_answer - len(response_list))
            response_list.extend(response_list_2)

        answer = "\n".join(response_list)

    temp_answer = process_answers(answer,question)
    final_result = {
        'question_id': None,  # 根据实际情况设置
        'question': question,
        'entity': entity,
        'final_ans1': answer,
        'ans_list': response_list,
        'final_ans': temp_answer
    }

    return final_result
