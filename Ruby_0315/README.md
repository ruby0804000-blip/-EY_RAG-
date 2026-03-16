# EY RAG Project

本專案為資安法規問答系統的 RAG prototype。

## 目前完成內容
- 將法規文件整理為 chunks_dataset.csv
- 使用 BAAI/bge-m3 建立向量
- 使用 ChromaDB 建立向量資料庫
- 使用 Ollama + Qwen2.5:3b 進行 RAG 問答

## 主要檔案
- `build_rag_db.py`：將 chunks_dataset.csv 建立為 ChromaDB
- `check_db.py`：檢查資料庫中的 collections
- `rag_chat.py`：執行本地 RAG 問答

## 執行步驟
1. 安裝 Python 虛擬環境
2. 安裝 requirements.txt 套件
3. 準備 `chunks_dataset.csv`
4. 執行 `python build_rag_db.py`
5. 執行 `python rag_chat.py`

