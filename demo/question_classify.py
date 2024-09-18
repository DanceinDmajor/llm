import csv
import pandas as pd
import numpy as np
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig
from share import tokenizer
from share import model
import copy

company_file_dir = '/root/llm/files/AF0_pdf_to_company.csv'
company_file = pd.read_csv(company_file_dir, delimiter=",", header=0)
company_list = company_file['公司名称'].tolist()



print('模型加载完成')

prompt = """
    你是一个问题分类器。对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。以下是一些例子：

    问题：“在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。”
    回答：“基金股票数据库”
    
    问题：“XXXX股份有限公司变更设立时作为发起人的法人有哪些？”
    回答：“该公司的招股说明书”
    
    问题：“我想知道XXXXXX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。”
    回答：“基金股票数据库”
    
    问题：“XXXXXX股份有限公司2020年增资后的投后估值是多少？”
    回答：“该公司的招股说明书”
    
    问题：“根据XXXXXX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？”
    回答：“该公司的招股说明书”
    
    问题：“什么公司、在何时与XXXXXX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？”
    回答：“该公司的招股说明书”
    
    问题：“请帮我查询下股票代码为XXXXXX的股票在2021年内最高日收盘价是多少？”
    回答：“基金股票数据库”
    
    问题：“XXXXXX股份有限公司的中标里程覆盖率为多少？”
    回答：“该公司的招股说明书”
    
    问题：“根据中国证监会颁布的《上市公司行业分类指导》的规定，XXXXXX有限公司所属的行业大类、中类、小类是什么？”
    回答：“该公司的招股说明书”
    
    问题：“请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。”
    回答：“基金股票数据库”
    
    问题：“XXXXXX有限公司和合肥翰林是否按规定为员工缴纳了社会保险？”
    回答：“该公司的招股说明书”
    
    问题：“我想知道XXXXXX有限公司在2020年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”
    
    问题：“根据《CRCC产品认证实施规则》，《铁路产品认证证书》有效期为多久？XXXXXX有限公司取得 《铁路产品认证证书》后，至少多久需要接受一次监督？”
    回答：“该公司的招股说明书”
    
    问题：“我想知道XXXXXX基金管理有限公司在2019年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”
    
    问题：“请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。”
    回答：“基金股票数据库”
    
    问题：“我想知道XXXXXX有限公司在2019年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”
    
    问题：“我想知道股票XXXXXX在申万行业分类下的二级行业是什么？用最新的数据。”
    回答：“基金股票数据库”
    
    问题：“请帮我查询下股票代码为XXXXXX的股票在2019年内最高日收盘价是多少？”
    回答：“基金股票数据库”
    
    问题：“股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）”
    回答：“基金股票数据库”
    
    问题：“截至2009年底，中海达、南方测绘合计占有国产品牌销售额的多大比例？”
    回答：“该公司的招股说明书”
    
    问题：“截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？”
    回答：“该公司的招股说明书”
    
    问题：“股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）”
    回答：“基金股票数据库”

    根据上面提供的例子对以下问题进行分类。
    问题：“
    """


def classify_question(question):
    prompt1 = prompt + question + """？”"""
    response_new, history_new = model.chat(tokenizer, prompt1, history=None)
    if '招股说明书' in response_new and '股票数据库' not in response_new:
        temp_class = 'Text'
    elif '招股说明书' not in response_new and '股票数据库' in response_new:
        temp_class = 'SQL'
        for company_name in company_list:
            if company_name in question:
                temp_class = 'Text'
    else:
        temp_class = 'SQL'
        for company_name in company_list:
            if company_name in question:
                temp_class = 'Text'
                
    return temp_class



def process_single_question( question_text, question_class):
    # 从文件读取公司信息
    company_file_dir = '/root/llm/files/AF0_pdf_to_company.csv'
    company_file = pd.read_csv(company_file_dir, delimiter=",", header=0)

    company_data_csv_list = []
    company_index_list = []
    company_name_list = []

    for cyc in range(len(company_file)):
        company_name_list.append(company_file.iloc[cyc]['公司名称'])
        company_data_csv_list.append(company_file.iloc[cyc]['csv文件名'])
        temp_index_cp = tokenizer(company_file.iloc[cyc]['公司名称'])
        temp_index_cp = temp_index_cp['input_ids']
        company_index_list.append(temp_index_cp)

    tempw_entity = 'N_A'
    tempw_csv_name = 'N_A'

    if question_class == 'Text':
        temp_index_q = tokenizer(question_text)
        temp_index_q = temp_index_q['input_ids']
        q_cp_similarity_list = []
        for cyc2 in range(len(company_file)):
            temp_index_cp = company_index_list[cyc2]
            temp_simi = len(set(temp_index_cp) & set(temp_index_q)) / (len(set(temp_index_cp)) + len(set(temp_index_q)))
            q_cp_similarity_list.append(temp_simi)

        t = copy.deepcopy(q_cp_similarity_list)
        max_index = t.index(max(t))
        tempw_entity = company_name_list[max_index]
        tempw_csv_name = company_data_csv_list[max_index]

    elif question_class == 'SQL':
        pass

    else:
        find_its_name_flag = False
        for cyc_name in range(len(company_name_list)):
            if company_name_list[cyc_name] in question_text:
                tempw_entity = company_name_list[cyc_name]
                tempw_csv_name = company_data_csv_list[cyc_name]
                find_its_name_flag = True
                break

    result = {
        '问题': question_text,
        '分类': question_class,
        '对应实体': tempw_entity,
        'csv文件名': tempw_csv_name
    }

    return result

