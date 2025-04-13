import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 環境変数の読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain LLM 初期化
llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

# アプリ概要
st.title("LLM専門家チャットアプリ")
st.markdown("""
このアプリでは、入力した質問に対してLLMが専門家として回答します。  
以下から専門家の種類を選び、質問を入力してください。
""")

# 入力フォーム
user_input = st.text_input("質問を入力してください")
expert_choice = st.radio("専門家の種類を選んでください", ("AI研究者（A）", "健康アドバイザー（B）"))

# プロンプトの切り替え
def get_prompt_template(expert):
    if "A" in expert:
        return PromptTemplate(
            input_variables=["question"],
            template="あなたは最先端AIの専門家です。以下の質問に簡潔かつ専門的に答えてください。\n質問: {question}"
        )
    else:
        return PromptTemplate(
            input_variables=["question"],
            template="あなたは健康アドバイザーです。以下の質問にやさしく丁寧に答えてください。\n質問: {question}"
        )

# 回答生成関数
def generate_response(question, expert):
    prompt = get_prompt_template(expert)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question)

# 回答ボタン
if st.button("送信") and user_input:
    with st.spinner("専門家が考えています..."):
        answer = generate_response(user_input, expert_choice)
        st.success("回答:")
        st.write(answer)
