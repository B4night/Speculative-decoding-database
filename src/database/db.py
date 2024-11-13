from SPARQLWrapper import SPARQLWrapper, JSON
import json
import requests
from bs4 import BeautifulSoup
import time, pudb
import pandas as pd

max_uri = 3
max_abs_len = 200

# 读取 JSON 文件并创建问题到 SPARQL 查询的映射
with open('/home/feic/pjs/Speculative-decoding-database/data/combined.json') as f:
    data = json.load(f)
    question_to_sparql = {
        item['corrected_question']: item['sparql_query'] for item in data
    }

def LC_QuAD_query(sparql_query, abstract_needed=True):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        initial_results = sparql.query().convert()
    except Exception as e:
        return [f"Error running SPARQL query: {e}"]
    
    if 'boolean' in initial_results:
        if initial_results['boolean']:
            return "Yes"
        else:
            return "No"
    
    answers = []
    uri_list = []
    # 处理初始结果，分离 URI 和其他类型的值
    for result in initial_results["results"]["bindings"]:
        for var_name, value_data in result.items():
            value_type = value_data["type"]
            value = value_data["value"]
            if value_type == "uri":
                uri_list.append(value)
            elif value_type == "literal":
                datatype = value_data.get("datatype", "")
                lang = value_data.get("xml:lang", "")
                if datatype:
                    datatype = datatype.split("#")[-1]
                    answers.append(f"{var_name} (Literal, type={datatype}): {value}")
                elif lang:
                    answers.append(f"{var_name} (Literal, lang={lang}): {value}")
                else:
                    answers.append(f"{var_name} (Literal): {value}")
            elif value_type == "bnode":
                answers.append(f"{var_name} (Blank Node): {value}")
            elif value_type == "triple":
                subject = value_data["value"]["subject"]["value"]
                predicate = value_data["value"]["predicate"]["value"]
                obj = value_data["value"]["object"]["value"]
                answers.append(
                    f"{var_name} (Triple): Subject: {subject}, Predicate: {predicate}, Object: {obj}"
                )
            else:
                answers.append(f"{var_name} ({value_type}): {value}")
    
    # Control the number of URIs
    if len(uri_list) > max_uri:
        uri_list = uri_list[:max_uri]
    
    if uri_list:
        answers.clear()

    if abstract_needed and uri_list:
        # 构建一个新的 SPARQL 查询来获取摘要和标签，处理重定向
        uri_values = " ".join(f"<{uri}>" for uri in uri_list)
        abstract_query = f"""
        SELECT ?original_subject ?subject ?label ?abstract WHERE {{
            VALUES ?original_subject {{ {uri_values} }}
            ?original_subject (dbo:wikiPageRedirects|^dbo:wikiPageRedirects)* ?subject .
            OPTIONAL {{
                ?subject dbo:abstract ?abstract .
                FILTER (lang(?abstract) = 'en')
            }}
            OPTIONAL {{
                ?subject rdfs:label ?label .
                FILTER (lang(?label) = 'en')
            }}
        }}
        """
        # 运行摘要查询
        sparql.setQuery(abstract_query)
        try:
            abstract_results = sparql.query().convert()
        except Exception as e:
            return [f"Error fetching abstracts: {e}"]
        
        # 创建一个字典，将原始 URI 映射到摘要和标签
        resource_dict = {}
        for result in abstract_results["results"]["bindings"]:
            original_subject = result.get("original_subject", {}).get("value", "")
            abstract = result.get("abstract", {}).get("value", "")
            label = result.get("label", {}).get("value", "")
            # control the length of abstract
            if len(abstract) > max_abs_len:
                abstract = abstract[:max_abs_len] + '...'
            resource_dict[original_subject] = {"abstract": abstract, "label": label}
        
        # 将摘要添加到答案列表中
        for uri in uri_list:
            resource_info = resource_dict.get(uri, {})
            label = resource_info.get('label', uri.split('/')[-1])
            abstract = resource_info.get('abstract', '')
            if abstract:
                answers.append(f"{label}: {abstract}")
            else:
                # 如果没有摘要，请求 URL 并解析 'p' 标签（类名为 'lead'）
                try:
                    response = requests.get(uri)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    lead_paragraph = soup.find('p', class_='lead')
                    lead_text = lead_paragraph.get_text() if lead_paragraph else ""
                    if lead_text:
                        answers.append(f"{label}: {lead_text}")
                    else:
                        answers.append(f"{label}: No abstract or lead paragraph available.")
                except Exception as e:
                    answers.append(f"{label}: Error fetching content: {e}")
    else:
        # 如果不需要摘要，只返回标签或 URI
        for uri in uri_list:
            # 获取标签
            label = uri.split('/')[-1]
            answers.append(label)
    
    # if answers contain "callret-0 (typed-literal):", replace the prefix with ""
    answers = [answer.replace("callret-0 (typed-literal):", "") for answer in answers]
    
    # replace "uri (Literal, lang=en): " with ""
    answers = [answer.replace("uri (Literal, lang=en): ", "") for answer in answers]
    
    # The type of answer is list, convert it to string
    answers = '\n'.join(answers)
    return answers

cnt = 0

def LC_QuAD_query_test(_):
    global cnt
    if cnt >= len(data):
        return "No more questions.", ""
    question = data[cnt]['corrected_question']
    query = data[cnt]['sparql_query']
    print(f'Question: {question}\n')
    answer = LC_QuAD_query(query)
    cnt = cnt + 1
    return question, answer


        
            

# Store_question_answer()

# with open('/home/feic/pjs/Speculative-decoding-database/src/output/db_test_modify', 'w') as f:
#     while True:
#         question, answer = LC_QuAD_query_test("")
#         if question == 'No more questions.':
#             break
#         f.write(f"Question: {question}\n")
#         f.write(f"Answer: {answer}\n\n\n\n")
#         f.flush()