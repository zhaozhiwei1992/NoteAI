"""
fastapi dev main.py

查看文档: http://127.0.0.1:8000/docs

"""
from typing import Union
from fastapi import FastAPI
import SearchWithLLM

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/init")
def read_item():
    return "success"


@app.get("/api/question")
def read_item(q: Union[str, None] = None):
    # curl http://127.0.0.1:8000?q=你好
    # return {"q": q}
    search_result = SearchWithLLM.exec(q)
    return search_result
