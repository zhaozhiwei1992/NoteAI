from pymilvus import MilvusClient

client = MilvusClient(
    collection_name="qux",
    uri="http://localhost:19530",
    vector_field="float_vector",
    # pk_field= "id", # 如果要提供自己的 PK
    overwrite=True,
)

# 数据写入
data = [
    {
        "float_vector": [1, 2, 3],
        "id": 1,
        "text": "foo"
    },
    {
        "float_vector": [4, 5, 6],
        "id": 2,
        "text": "bar"
    },
    {
        "float_vector": [7, 8, 9],
        "id": 3,
        "text": "baz"
    }
]
client.insert(collection_name="qux", data=data)

# 查询
res = client.search(
    data=[[1, 3, 5], [7, 8, 9]],
    top_k=2,
)
print(res)
