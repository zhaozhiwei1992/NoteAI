"""
加载到内存处理
"""
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
import os
import datetime
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取配置文件
with open('config.json', 'r') as f:
    config = json.load(f)

def exec():
    # 检查collection是否存在, 如果不指定，默认为LangChainCollection
    collection_name = config.get("collection_name", "note")
    # 设置 Milvus 客户端
    client = MilvusClient(uri=config.get("milvus_uri", "http://localhost:19530"))
    # 检查collection是否存在
    if client.has_collection(collection_name):
        # collection存在，执行后续操作
        logging.INFO(f"Collection '{collection_name}' exists.")
    else:
        # collection不存在，创建collection并进行向量化
        print(f"Collection '{collection_name}' does not exist. Creating now...")
        # 从url导入知识作为聊天背景上下文, glob代表只查找org文件，可根据实际情况调整为txt等，recursive=True表示会递归查找
        loader = DirectoryLoader(os.path.join(os.environ["HOME"], "Documents/notes"), glob="*.org", recursive=True)
        # 加载一堆文件
        docs = loader.load()

        # 文本分词器
        # chunk_size=1000
        # 表示拆分的文档的大小，也就是上面所说的要设置为多少合适取决于所使用LLM 的窗口大小
        # chunk_overlap=100
        # 这个参数表示每个拆分好的文档重复多少个字符串。
        # 不过这种递归的方式更只能点，不设参数试试默认
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)

        # ollama嵌入层
        embeddings = OllamaEmbeddings(
            model="llama3.2:3b"
        )

        # 文档向量化，会持久化
        vector_store = Milvus.from_documents(documents=documents, embedding=embeddings, collection_name=collection_name,
                                             drop_old=True)
        print(f"collection'{collection_name}'创建成功!")


if __name__ == '__main__':

    start_time = datetime.datetime.now()
    exec()
    end_time = datetime.datetime.now()
    print(f"执行时间：{end_time - start_time}")