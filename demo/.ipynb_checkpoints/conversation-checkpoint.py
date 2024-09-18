import sys
import sqlite3
import re
import copy
import pandas as pd
from collections import Counter
import math
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from share import tokenizer
from share import model

from question_classify import classify_question, process_single_question
from data_quary import execute_sql_query, generate_sql_for_question, generate_answer
from text_question_answer import find_relevant_pieces, find_relevant_pieces_from_pdf, generate_final_answers


def main():
    while True:
        question = input("请输入您的问题：")
        if question.lower() in ['exit', 'quit']:
            break
        question_class = classify_question(question)
        print(f"问题分类结果: {question_class}")

        # 根据分类结果处理问题
        if question_class == 'SQL':
            # 调用SQL查询处理函数
            print("生成sql")
            r1 = generate_sql_for_question(question)
            sql = r1.get('SQL语句')
            print(sql)
            print("执行sql")
            sql_result = execute_sql_query(question, sql).get("执行结果")
            print(sql_result)
            print("生产答案中...")
            result = generate_answer(question, sql_result)
            print(result.get('FA'))
        elif question_class == 'Text':
            print("寻找实体中...")
            entity = process_single_question(question, 'Text')
            print(entity)
            print("计算相关度1...")
            relevant1 = find_relevant_pieces(entity.get('问题'), entity.get('对应实体'), entity.get('csv文件名'))
            print("计算相关度2...")
            relevant2 = find_relevant_pieces_from_pdf(entity.get('问题'), entity.get('对应实体'),
                                                      entity.get('csv文件名'))
            result = generate_final_answers(question, entity, relevant1.get('top_n_pages'),
                                            relevant2.get('top_n_pages'))
            print("综合的答案:")
            print(result.get('final_ans'))
            print("可能的答案召回文段")
            print(result.get('final_ans1'))
    
        else:
            print("无法识别的问题类型，请重试。")


if __name__ == "__main__":
    main()









