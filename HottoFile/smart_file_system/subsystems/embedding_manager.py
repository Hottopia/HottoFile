import os
import subprocess
import faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingManager:
    def __init__(self, persist_path="faiss_index"):
        """
        自动检测 bge-m3 是否在 Ollama 本地可用
        persist_path: 保存/加载向量数据库的路径
        """
        self.persist_path = persist_path
        self.embeddings = self._init_embeddings()
        self.vectorstore = None

        # 尝试加载已有索引
        if os.path.exists(self.persist_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ 已加载本地索引：{self.persist_path}")
            except Exception as e:
                print(f"⚠️ 无法加载已有索引，重新建立: {e}")

    def _init_embeddings(self):
        """检测并初始化 Embedding 模型"""
        if self._check_ollama_model("bge-m3"):
            print("✅ 检测到 Ollama 本地有 bge-m3，使用 OllamaEmbeddings")
            return OllamaEmbeddings(model="bge-m3")
        else:
            print("⚠️ 未检测到 Ollama bge-m3，使用 HuggingFace BGE-M3（CPU模式）")
            return HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},  # 如果有 CUDA 可改成 "cuda"
                encode_kwargs={"normalize_embeddings": True}
            )

    def _check_ollama_model(self, model_name):
        """检查 Ollama 是否有某个模型"""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            return model_name in result.stdout
        except Exception as e:
            print(f"⚠️ 无法检测 Ollama 模型: {e}")
            return False

    def build_index(self, file_objects):
        """用文件对象（FileInfo）列表建立索引"""
        texts = []
        metadatas = []
        for f in file_objects:
            if f.content["text_summary"]:
                texts.append(f.content["text_summary"])
                metadatas.append(f.to_dict())

        if texts:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            self.save_index()

    def add_file(self, file_obj):
        """单文件添加到向量数据库"""
        if not self.vectorstore:
            self.build_index([file_obj])
            return

        if file_obj.content["text_summary"]:
            self.vectorstore.add_texts(
                [file_obj.content["text_summary"]],
                metadatas=[file_obj.to_dict()]
            )
            self.save_index()

    def delete_file(self, file_id):
        """删除指定 file_id 的文件"""
        if not self.vectorstore:
            return False

        try:
            self.vectorstore.docstore._dict = {
                k: v for k, v in self.vectorstore.docstore._dict.items()
                if v.metadata["file_id"] != file_id
            }
            self.vectorstore.index.reset()
            self.vectorstore.add_texts(
                [doc.page_content for doc in self.vectorstore.docstore._dict.values()],
                metadatas=[doc.metadata for doc in self.vectorstore.docstore._dict.values()]
            )
            self.save_index()
            print(f"🗑️ 已删除文件: {file_id}")
            return True
        except Exception as e:
            print(f"⚠️ 删除失败: {e}")
            return False

    def update_file(self, file_obj):
        """更新文件（先删再加）"""
        deleted = self.delete_file(file_obj.file_id)
        self.add_file(file_obj)
        if deleted:
            print(f"🔄 文件已更新: {file_obj.name}")
        else:
            print(f"➕ 文件已新增: {file_obj.name}")

    def search(self, query, k=3):
        """搜索"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def save_index(self):
        """保存向量数据库"""
        if self.vectorstore:
            self.vectorstore.save_local(self.persist_path)
            print(f"💾 索引已保存到 {self.persist_path}")
