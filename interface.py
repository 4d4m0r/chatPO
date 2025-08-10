import gradio as gr
from llama_index.core.memory import ChatSummaryMemoryBuffer
from config import CONFIG
from components import *
from pipeline import *
import pandas as pd

chat_engine = initialize_chatbot()

def answer_llm(chat_engine, csv_path, output_csv_path=None):
    print(f"Lendo perguntas do arquivo CSV: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8")

    respostas = []
    for pergunta in df["question"]:
        print(f"Pergunta: {pergunta}")
        print("Enviando pergunta para o LLM...")
        resposta_llm = chat_engine.chat(pergunta).response
        print(f"Resposta do LLM: {resposta_llm}")
        respostas.append(resposta_llm)

    df["resposta_llm"] = respostas

    output_path = output_csv_path if output_csv_path else csv_path
    
    print("Escrevendo respostas no arquivo CSV...")
    df.to_csv(output_path, index=False, encoding="utf-8")

# Executa
answer_llm(chat_engine, CONFIG["input_csv_path"], CONFIG["output_csv_path"])