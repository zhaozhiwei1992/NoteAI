"""
使用pymilvus测试milvus库
"""
# 参考: https://blog.csdn.net/hello_dear_you/article/details/127841589
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# 创建collection
connections.connect("default", host="localhost", port="19530")

# filed可以类比为表示数据库中的列
# 定义了集合（Collection）的 schema三个字段：一个主键字段 pk，一个类型为双精度浮点数的字段 random，和一个类型为浮点数向量的字段 embeddings，向量维度为 8
fields = [
    # pk 类似mysql中的id, dtype类似mysql中的type, is_primary=True表示主键
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]
schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
# 使用定义的 schema 创建一个名为 hello_milvus 的集合。
hello_milvus = Collection("hello_milvus", schema)

# 构建数据并插入到collection中
import random

# entities: 可以类比为关系数据库写入的数据，只不过关系库是按行写，这里是按列写入
entities = [
    [i for i in range(3000)],  # field pk
    [float(random.randrange(-20, -10)) for _ in range(3000)],  # field random
    [[random.random() for _ in range(8)] for _ in range(3000)],  # field embeddings
]
insert_result = hello_milvus.insert(entities)

# 为实例创建索引, 为 embeddings 字段创建一个索引，以优化搜索性能。这里使用的是 IVF_FLAT 索引类型，L2 距离度量，并且参数 nlist 设置为 128。
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
hello_milvus.create_index("embeddings", index)

# 将collection中的数据加载到内存中执行向量相似度检索
hello_milvus.load()
vectors_to_search = entities[-1][-2:]
# vectors_to_search = [[0.4348870857133409, 0.190657166600133, 0.5978610871971484, 0.38415348295018603, 0.5988018724700892, 0.33483613584349026, 0.4243896851010588, 0.07090552061706146]]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])
print(result)

# 普通搜索
result = hello_milvus.query(expr="random > -14", output_fields=["random", "embeddings"])
print(result)

# 复合搜索
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > -12",
                             output_fields=["random"])

print(result)

# 删除collection
utility.drop_collection("hello_milvus")
