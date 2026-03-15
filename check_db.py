import chromadb

client = chromadb.PersistentClient(path=r"C:\Users\ruby0\Desktop\EY_RAG\chroma_db")

print("目前所有 collections：")
print(client.list_collections())