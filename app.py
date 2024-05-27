import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings("ignore")

# Api Key:
GROQ_API_KEY = "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"

# Modelo de LLM - Mixtral:
llm = ChatGroq(temperature=0.7,
               groq_api_key=GROQ_API_KEY,
               model_name="mixtral-8x7b-32768")

# App streamlit:
st.set_page_config(page_title="Assistente para análise de dados",
                   page_icon=":robot_face:",
                   layout="centered")

st.header(":male-technologist: Assistente IA | Análise de dados | DFS :male-technologist:")

with st.expander("Apresentação do assistente de análise de dados"):

  st.markdown('''
  O app permite importar um arquivo CSV e fazer perguntas para o assistente de dados um expert
  em análise de dados com python e pandas. Analise seus dados, extraia insights e
  tome decisões informadas com facilidade e agilizando processos de tomada de decisão.
  ''')

  st.markdown('''
  **Agente de IA para análise de dados: Pandas AI**

  O PandasAI é uma biblioteca que integra inteligência artificial com pandas, permitindo a 
  análise e manipulação de dados de forma mais intuitiva e eficiente. Utiliza modelos de 
  linguagem para interpretar comandos em linguagem natural, automatizando tarefas complexas 
  de análise de dados.
  
  ''')

  st.markdown('''
  **Modelo LLM: Mixtral - 8x7b**

  O modelo Mixtral é um modelo de linguagem natural multilíngue desenvolvido pela LightOn.
  Ele é projetado para entender e gerar texto em várias línguas, sendo treinado com grandes
  quantidades de dados diversos. O Mixtral é conhecido por sua capacidade de transferir
  conhecimentos entre línguas diferentes, melhorando a qualidade das traduções e outras
  tarefas de processamento de linguagem natural em cenários multilíngues.
  ''')

csv_file = st.file_uploader("Carregue o seu arquivo csv:", type="csv")

if csv_file is not None:
    data = pd.read_csv(csv_file,sep=",")
    data = data.infer_objects()
    df = SmartDataframe(data,config={"llm": llm})
    st.write("Previa dos seus dados:")
    st.dataframe(data.head())

    # check for messages in session and create if not exists
    if "messages" not in st.session_state.keys():
      st.session_state.messages = [{"role": "assistant", "content": "Sou um assistente para análise de dados, como posso ajudar?"}]

    # Display all messages
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.write(message["content"])

    user_prompt = st.chat_input()
    
    if user_prompt is not None:
      st.session_state.messages.append({"role": "user", "content": user_prompt})
      
      with st.chat_message("user"):
        st.write(user_prompt)
    
    if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          with st.spinner("Loading..."):
              ai_response = df.chat(user_prompt)
              st.write(ai_response)
              st.set_option('deprecation.showPyplotGlobalUse',False)
              st.pyplot()
    
      new_ai_message = {"role": "assistant", "content": ai_response}
      st.session_state.messages.append(new_ai_message)

else:
    st.info('O sistema está ligado!', icon="ℹ️")
