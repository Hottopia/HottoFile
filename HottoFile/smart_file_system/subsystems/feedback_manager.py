import json
import os

class FeedbackManager:
    def __init__(self, storage_path="feedback.json", embedding_manager=None):
        """
        Args:
            storage_path (str): JSON file to store feedback
            embedding_manager: Optional, if provided will update FAISS when feedback is added
        """
        self.storage_path = storage_path
        self.embedding_manager = embedding_manager

        if os.path.exists(storage_path):
            with open(storage_path, "r", encoding="utf-8") as f:
                self.corrections = json.load(f)
        else:
            self.corrections = []

    def add_feedback(self, file_info, correct_label):
        """
        Add feedback for a file, persist it, and update embedding DB if provided.
        """
        file_info.update_label(correct_label, feedback=True)

        correction = {
            "file_id": file_info.file_id,
            "name": file_info.name,
            "summary": file_info.content.get("text_summary", ""),
            "correct_label": correct_label
        }

        # 存到内存
        self.corrections.append(correction)

        # 存到 JSON 文件
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.corrections, f, indent=4, ensure_ascii=False)

        # 如果有 embedding_manager，更新向量数据库
        if self.embedding_manager:
            try:
                self.embedding_manager.add_file(file_info)
                print(f"🔄 Embedding DB updated for corrected file: {file_info.name}")
            except Exception as e:
                print(f"⚠️ Could not update embedding DB: {e}")

        print(f"✅ Feedback saved for {file_info.name} -> {correct_label}")

    def load_feedback(self):
        """Return all stored corrections"""
        return self.corrections

    def get_fewshot_examples(self, n=3):
        """Get a few feedback examples for LLM few-shot prompting"""
        return self.corrections[-n:] if self.corrections else []

