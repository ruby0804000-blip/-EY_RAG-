import chromadb
from sentence_transformers import SentenceTransformer
import requests

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3")

# 一定要用和 build_rag_db.py 一樣的資料庫位置
client = chromadb.PersistentClient(path=r"C:\Users\ruby0\Desktop\EY_RAG\chroma_db")
collection = client.get_collection(name="ey_rag_chunks")

print("資料庫載入完成")

while True:
    question = input("\n請輸入問題（exit離開）： ")

    if question.lower() == "exit":
        break

    # 問題轉向量
    q_embedding = model.encode([question]).tolist()

    # 查詢最相關的 3 段
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3
    )

    docs = results["documents"][0]
    context = "\n\n".join(docs)

    prompt = f"""
你是一個「資安法規問答助手」。

你的任務是根據系統提供的法規內容，回答使用者的問題。

請嚴格遵守以下規則：
1. 只能根據「參考法規內容」回答，不可使用外部知識，不可自行推測，不可補充未出現在法規內容中的資訊。
2. 若參考法規內容不足以回答問題，或與問題無直接相關，請直接回答：
「抱歉，我們的法規庫中沒有相關資訊，無法提供答案。」
3. 回答一律使用繁體中文。
4. 回答應清楚、正式、精簡。
5. 不要編造法條名稱、不要編造條號、不要假裝知道未提供的內容。

【使用者問題】
{question}

【參考法規內容】
{context}

請開始回答：
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    print("\n回答：")
    print(answer)
    