"""
使用Rag技术实现本地知识库，使用ollama, 结合milvus向量库
"""
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient


def exec(question):
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
        # 创建ollama 模型 llama2
        llm = OllamaLLM(model="llama3.2:3b")
        output_parser = StrOutputParser()

        # 创建提示词模版
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}"""
        )

        # 生成chain ：   prompt | llm
        document_chain = create_stuff_documents_chain(llm, prompt)

        # 向量数据库检索器
        retriever = vector_store.as_retriever()

        # 向量数据库检索chain :  vector | prompt | llm
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # 调用上面的 (向量数据库检索chain)
        response = retrieval_chain.invoke({"input": question})
        # 打印结果
        print(response["answer"])


if __name__ == '__main__':
    exec("我有什么梦想? 如何实现")
