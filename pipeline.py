from components import *
from config import CONFIG
from llama_index.core.memory import ChatSummaryMemoryBuffer

def initialize_chatbot():
    print("Inicializando o chatbot...")
    docs = load_documents(CONFIG["docs_path"])
    nodes = parse_nodes(docs, CONFIG["chunk_size"])

    print("Inicializando o banco de dados Chroma...")
    embed_wrapper = ChromaEmbeddingWrapper(CONFIG["embedding_model"])
    chroma_collection = setup_chroma_db(CONFIG["chroma_path"], CONFIG["collection_name"], embed_wrapper)

    print("Construindo o índice...")
    index, storage_context, embed_model = build_index(nodes, chroma_collection, CONFIG["embedding_model"])

    print("Configurando o LLM...")
    llm = setup_llm(CONFIG["llm_model"])
    memory = ChatSummaryMemoryBuffer(llm=llm, token_limit=CONFIG["chat_token_limit"])

    print("Configurando o mecanismo de chat...")
    chat_engine = setup_chat_engine(index, llm, memory, CONFIG["chat_system_prompt"])

    # print("Teste do chatbot...")
    # response = chat_engine.chat('Olá')
    # print(response)
    return chat_engine