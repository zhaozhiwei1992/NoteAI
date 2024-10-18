# NoteAI
本地笔记AI

# 概述
通过AI整合笔记，充分发掘个人笔记的价值

# 环境

- docker27.3.1、docker-compose
- 向量库: Milvus
- python3.12.7
- 系统: archlinux
- 模型: ollama + llama3.2:3b

# 运行
1. 准备好上述环境，python最好通过conda做 虚拟环境
2. 初始化向量库
    ```
    python LoadFile2Vector.py
    ```
3. 对外提供接口
    ```
    fastapi dev Main.py
    ```
    后续便可以通过接口将知识库与自己的项目进行对接。

