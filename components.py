import chromadb
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

class ChromaEmbeddingWrapper:
    def __init__(self, model_name):
        self.model = HuggingFaceEmbedding(model_name=model_name)

    def __call__(self, input):
        return self.model.embed(input)

    def name(self):
        return "ChromaE5Wrapper"

def load_documents(path):
    reader = SimpleDirectoryReader(input_dir="pdf_database")
    return reader.load_data()

def parse_nodes(docs, chunk_size):
    parser = SentenceSplitter(chunk_size=chunk_size)
    return parser.get_nodes_from_documents(docs, show_progress=True)

def setup_chroma_db(path, collection_name, embedding_fn):
    client = chromadb.PersistentClient(path=path)
    try:
        return client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)
    except Exception as e:
        raise RuntimeError(f"Erro ao criar/get coleção do Chroma: {e}")

def build_index(nodes, collection, embedding_model_name):
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    return VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model), storage_context, embed_model

def setup_llm(model_name):
    return Ollama(model=model_name, request_timeout=120.0, context_window=8000)

def setup_chat_engine(index, llm, memory, prompt):
    return index.as_chat_engine(
        chat_mode='context',
        llm=llm,
        memory=memory,
        system_prompt=prompt
    )
