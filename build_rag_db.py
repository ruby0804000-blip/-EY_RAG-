import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("開始讀取 CSV...")
csv_path = r"C:\Users\ruby0\Desktop\EY_RAG\chunks_dataset.csv"
df = pd.read_csv(csv_path)
print("資料筆數：", len(df))

# 先取前 600 筆做測試
df_small = df.head(600).copy()
print("本次建立資料庫筆數：", len(df_small))

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3")
print("Embedding 模型載入完成")

# 建立 Persistent ChromaDB
db_path = r"C:\Users\ruby0\Desktop\EY_RAG\chroma_db"
client = chromadb.PersistentClient(path=db_path)

# 如果舊的 collection 存在就刪掉
try:
    client.delete_collection(name="ey_rag_chunks")
    print("舊的 ey_rag_chunks 已刪除")
except:
    print("沒有舊的 ey_rag_chunks，直接建立新的")

# 這裡一定要是 create_collection，不是 get_collection
collection = client.create_collection(name="ey_rag_chunks")
print("ChromaDB collection 建立完成")

# 準備資料
texts = df_small["text"].fillna("").tolist()
sources = df_small["source"].astype(str).tolist()
chunk_ids = df_small["chunk_id"].astype(str).tolist()
ids = [f"{sources[i]}_chunk_{chunk_ids[i]}" for i in range(len(df_small))]

print("開始生成 embeddings...")
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True
)

print("開始寫入 ChromaDB...")
collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[
        {"source": sources[i], "chunk_id": chunk_ids[i]}
        for i in range(len(df_small))
    ],
    ids=ids
)

print("完成！")
print("目前資料筆數：", collection.count())
print("資料庫位置：", db_path)