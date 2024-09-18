import jsonlines


# 读取一个.jsonl文件并将其内容转换为Python列表
def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


# 将一个Python列表的内容写入到一个.jsonl文件中
def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)


