import sqlite3
import re
import copy
from share import tokenizer
from share import model
import pandas as pd
# 定义表名列表
table_name_list = ['基金基本信息','基金股票持仓明细','基金债券持仓明细','基金可转债持仓明细','基金日行情表','A股票日行情表','港股票日行情表','A股公司行业划分表','基金规模变动表','基金份额持有人结构']
term_list_1 = ['基金股票持仓明细','基金债券持仓明细','基金可转债持仓明细']
n = 5  # 定义相似问题的数量

# 定义一些全局变量
deny_list = ['0','1','2','3','4','5','6','7','8','9','，','？','。', '一','二','三','四','五','六','七','八','九','零','十', '的','小','请','.','?','有多少','帮我','我想','知道', '是多少','保留','是什么','-','(',')','（','）','：', '哪个','统计','且','和','来','请问','记得','有','它们']
pattern1 = r'\d{8}'

# 创建SQL数据库连接
conn = sqlite3.connect('/root/llm/data/dataset/stockdata.db',check_same_thread=False)
cs = conn.cursor()




# 定义一个生成sql prompt
def get_prompt_v33(question, index_list):
    Examples = '以下是一些例子：'
    for index in index_list:
        Examples = Examples + "问题：" + example_question_list[index] + '\n'
        Examples = Examples + "SQL：" + example_sql_list[index] + '\n'

    impt2 = "你是一个精通SQL语句的程序员。我会给你一个问题，请按照问题描述，仿照以下例子写出正确的SQL代码。"
    impt2 = impt2 + Examples
    impt2 = impt2 + "问题：" + question + '\n'
    impt2 = impt2 + "SQL："
    return impt2


# 从CSV文件加载示例问题和SQL
SQL_examples_file_dir = "/root/llm/files/ICL_EXP.csv"
SQL_examples_file = pd.read_csv(SQL_examples_file_dir, delimiter=",", header=0)
example_question_list = SQL_examples_file['问题'].tolist()
example_sql_list = SQL_examples_file['SQL'].tolist()
example_token_list = [tokenizer(question)['input_ids'] for question in example_question_list]

# 生成用于忽略的token列表
deny_token_list = []
for word in deny_list:
    deny_token_list += tokenizer(word)['input_ids']

# 生成sql
# 处理单个问题并生成SQL查询语句
def generate_sql_for_question(user_question):
    response2 = 'N_A'
    prompt2 = 'N_A'

    temp_question = user_question
    date_list = re.findall(pattern1, temp_question)
    temp_question2_for_search = temp_question
    for t_date in date_list:
        temp_question2_for_search = temp_question2_for_search.replace(t_date, ' ')
    temp_tokens = tokenizer(temp_question2_for_search)
    temp_tokens = temp_tokens['input_ids']
    temp_tokens2 = [x for x in temp_tokens if x not in deny_token_list]
    temp_tokens = temp_tokens2

    # 计算与已有问题的相似度
    similarity_list = list()
    for cyc2 in range(len(SQL_examples_file)):
        similarity_list.append(len(set(temp_tokens) & set(example_token_list[cyc2])) / (
                    len(set(temp_tokens)) + len(set(example_token_list[cyc2]))))

    # 求与第X个问题相似的问题
    t = copy.deepcopy(similarity_list)
    # 求n个最大的数值及其索引
    max_number = []
    max_index = []
    for _ in range(n):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)
    t = []

    temp_length_test = ""
    short_index_list = list()
    for index in max_index:
        temp_length_test_1 = temp_length_test
        temp_length_test = temp_length_test + example_question_list[index]
        temp_length_test = temp_length_test + example_sql_list[index]
        if len(temp_length_test) > 2300:
            break
        short_index_list.append(index)

    prompt2 = get_prompt_v33(user_question, short_index_list)
    response2, history = model.chat(tokenizer, prompt2, history=None)

    # 保存最终结果到变量
    result = {
        "问题": user_question,
        "SQL语句": response2,
        "prompt": prompt2
    }

    return result



def execute_sql_query(question,sql_query):
    # 替换表名的映射
    replacement_dict = {
        "B股票日行情表": "A股票日行情表",
        "创业板日行情表": "A股票日行情表",
        " 股票日行情表": " A股票日行情表",
        " 港股日行情表": " 港股票日行情表"
    }
    # 类似表名的列表
    term_list_1 = ['基金股票持仓明细', '基金债券持仓明细', '基金可转债持仓明细']

    # 替换表名以提高查询成功率
    for old_table, new_table in replacement_dict.items():
        sql_query = sql_query.replace(old_table, new_table)

    # 去除可能影响查询的特殊字符
    sql_query = sql_query.replace("”", '').replace("“", '')

    origin_success_flag = 0
    SQL_exe_result = 'N_A'
    Use_similar_table_flag = 0
    SQL_exe_flag = 0

    try:
        # 尝试执行原始SQL查询
        cs.execute(sql_query)
        cols = cs.fetchall()
        SQL_exe_result = str(cols)
        origin_success_flag = 1
    except Exception as e:
        SQL_exe_result = str(e)
        # 如果原始查询失败，尝试使用相似表名
        for item in term_list_1:
            if item in sql_query:
                Use_similar_table_flag = 1
                original_item = item
                break

        if Use_similar_table_flag == 1:
            for item in term_list_1:
                try:
                    # 尝试替换表名并重新执行查询
                    cs.execute(sql_query.replace(original_item, item))
                    cols = cs.fetchall()
                    SQL_exe_result = str(cols)
                    SQL_exe_flag = 2
                    origin_success_flag = 1
                    break
                except Exception as e:
                    SQL_exe_result = str(e)
                    continue

    # 如果所有尝试都失败，记录执行失败信息
    if origin_success_flag == 0:
        SQL_exe_result = "执行失败: " + SQL_exe_result

    result = {
        '问题': question,
        '能否成功执行': origin_success_flag,
        '执行结果': SQL_exe_result,
        'List': term_list_1 if Use_similar_table_flag else [] # 没有使用相似表名进行替换，这个字段将为空列表。
    }
    return result


# 定义根据sql结果生成答案Prompt
def get_prompt_v34(question, data, index_list):
    # 读取示例数据
    SQL_examples_file_dir = "/root/llm/files/ICL_EXP.csv"
    SQL_examples_file = pd.read_csv(SQL_examples_file_dir, delimiter=",", header=0)

    example_question_list = list()
    example_data_list = list()
    example_FA_list = list()
    example_token_list = list()

    for cyc in range(len(SQL_examples_file)):
        example_question_list.append(SQL_examples_file[cyc:cyc + 1]['问题'][cyc])
        example_data_list.append(SQL_examples_file[cyc:cyc + 1]['资料'][cyc])
        example_FA_list.append(SQL_examples_file[cyc:cyc + 1]['FA'][cyc])
        temp_tokens = tokenizer(SQL_examples_file[cyc:cyc + 1]['问题'][cyc])
        temp_tokens = temp_tokens['input_ids']
        temp_tokens2 = [x for x in temp_tokens if x not in deny_token_list]
        example_token_list.append(temp_tokens2)


    Examples = '以下是一些例子：'
    for index in index_list:
        Examples = Examples + "问题：" + example_question_list[index] + '\n'
        Examples = Examples + "资料：" + example_data_list[index] + '\n'
        Examples = Examples + "答案：" + example_FA_list[index] + '\n'
    impt2 = """
        你要进行句子生成工作，根据提供的资料来回答对应的问题。下面是一些例子。注意问题中对小数位数的要求。+ '\n'
    """
    impt2 = impt2 + Examples
    impt2 = impt2 + "问题：" + question + '\n'
    impt2 = impt2 + "资料：" + data + '\n'
    impt2 = impt2 + "答案："
    return impt2
# 根据sql结果生成答案
def generate_answer(user_question,user_sql_result):
    temp_FA = user_question

    if user_sql_result != 'N_A':
        if len(user_sql_result) > 0:
            if len(user_sql_result) > 250:
                user_sql_result = user_sql_result[0:250]
            date_list = re.findall(pattern1, user_question)
            temp_question2_for_search = user_question
            for t_date in date_list:
                temp_question2_for_search = temp_question2_for_search.replace(t_date, ' ')
            temp_tokens = tokenizer(temp_question2_for_search)
            temp_tokens = temp_tokens['input_ids']
            temp_tokens2 = [x for x in temp_tokens if x not in deny_token_list]
            temp_tokens = temp_tokens2

            # 计算与已有问题的相似度
            similarity_list = list()
            for cyc2 in range(len(SQL_examples_file)):
                similarity_list.append(len(set(temp_tokens) & set(example_token_list[cyc2])) / (
                            len(set(temp_tokens)) + len(set(example_token_list[cyc2]))))

            # 求与第X个问题相似的问题
            t = copy.deepcopy(similarity_list)
            # 求n个最大的数值及其索引
            max_number = []
            max_index = []
            for _ in range(n):
                number = max(t)
                index = t.index(number)
                t[index] = 0
                max_number.append(number)
                max_index.append(index)
            t = []

            prompt2 = get_prompt_v34(user_question, user_sql_result, max_index)
            temp_FA, history = model.chat(tokenizer, prompt2, history=None)
    else:
        user_sql_result = 'SQL未能成功执行！'

    # 保存最终结果到变量
    result = {
        "问题": user_question,
        "FA": temp_FA,
        "SQL结果": user_sql_result
    }

    return result