"""
使用pymilvus测试milvus库
"""
# 参考: https://blog.csdn.net/hello_dear_you/article/details/127841589
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

flag = client.has_collection(collection_name="note")

print(flag)

client.drop_collection(collection_name="LangChainCollection")