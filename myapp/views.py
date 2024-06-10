from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import openai
from dotenv import load_dotenv
import requests
import streamlit as st
from streamlit_chat import message

load_dotenv()
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

datapath = r"csv\culture_embedding.csv"
df = pd.read_csv(datapath, index_col=0)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

missing_values_count = df.isna().sum()
if missing_values_count.sum() > 0:
    df.fillna(" ", inplace=True)
    df.to_csv(datapath)

if 'combined' not in df.columns:
    df["combined"] = (
        df['과목명'].str.strip() + " / " + df['학수번호'].str.strip() + " / " + df['분반'].astype(str).str.strip() +
        " / " + df['학점'].astype(str).str.strip() + " / " + df['담당교수'].str.strip() + " / " + df['강의시간'].str.strip() +
        " / " + df['종류'].str.strip() + " / " + df['영역'].str.strip()
    )
    df.to_csv("culture_combined.csv", index=False, encoding='utf-8-sig')

if 'embedding' not in df.columns:
    df['embedding'] = df.apply(lambda row: get_embedding(row.combined), axis=1)
    df.to_csv("culture_embedding.csv", index=False, encoding='utf-8-sig')

def get_embedding(text):
    response = client.embeddings.create(
        input = text,
        model = 'text-embedding-ada-002'
    )
    return response.data[0].embedding

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    df['similarity'] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(5)
    return top_three_doc

def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    system_role = f"""You are an AI language model that answers questions
    And you answer documents about open subjects
    You must import the specified document and return the contents of the document in the query language
    과목명 / 학수번호 / 분반 / 학점 / 담당교수 / 강의시간 / 종류 / 영역 Answer in the same format.
    The subject name must be included in the answer.
    Here are the document: 
            doc 1 :{str(result.iloc[0])}
            doc 2 :{str(result.iloc[1])}
            doc 3 :{str(result.iloc[2])}
            doc 4 :{str(result.iloc[3])}
            doc 5 :{str(result.iloc[4])}
    You must return in Korean. Return a accurate answer based on the document.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]
    return messages

def generate_response(messages):
    result = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::9QEaezHG",
        messages=messages,
        temperature=0.5,
        max_tokens=500)
    return result.choices[0].message.content

def parse_response(response):
    try:
        if '/' in response:
            parsed_elements = response.split(' / ')
        else:
            parsed_elements = response.split(', ')
        response_dict = {
            "과목명": parsed_elements[0].strip(),
            "학수번호": parsed_elements[1].strip(),
            "분반": parsed_elements[2].strip(),
            "학점": parsed_elements[3].strip(),
            "담당교수": parsed_elements[4].strip(),
            "강의시간": parsed_elements[5].strip(),
            "종류": parsed_elements[6].strip()
        }
        if len(parsed_elements) == 8:
            response_dict["영역"] = parsed_elements[7].strip()

        return response_dict
    except IndexError:
        return {"error": "Parsing error", "response": response}

@csrf_exempt
def send_query(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received data:", data)
            query = data.get('query', '').strip()
            print("Query: ", query)

            messages = create_prompt(df, query)
            response = generate_response(messages)
            response_dict = parse_response(response)
            return JsonResponse(response_dict, json_dumps_params={'ensure_ascii': False})
        except json.JSONDecodeError as e:
            print("JSON decode error:", str(e))
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
