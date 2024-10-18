"""
请先执行LoadFile2Vector
"""
from langchain_community.vectorstores import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

# 检查collection是否存在
collection_name = "note"

# 设置 Milvus 客户端
client = MilvusClient(uri="http://localhost:19530")

vector_store = None

# 检查collection是否存在
if client.has_collection(collection_name):
    # collection存在，执行后续操作
    print(f"Collection '{collection_name}' exists.")

    # 文本分词器
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents([])

    # ollama嵌入层
    embeddings = OllamaEmbeddings(
        model="llama3.2:3b"
    )
    # 文档向量化
    vector_store = Milvus.from_documents(documents=documents, embedding=embeddings, collection_name=collection_name)
else:
    # collection不存在，创建collection并进行向量化
    print(f"Collection '{collection_name}' does not exist. Please exec LoadFile2Vector.py first")

if vector_store is not None:
    query = "我的梦想是"
    docs = vector_store.similarity_search(query)
    print(docs)