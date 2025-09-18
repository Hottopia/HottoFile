
import json
import uuid
from datetime import datetime
from datetime import datetime

class FileInfo:
    def __init__(self, path, name, ext, ftype, size, created_at=None, modified_at=None, 
                 text_summary=None, image_features=None, audio_transcript=None, video_keyframes=None):
        # 基础信息
        self.file_id = str(uuid.uuid4())  # 自动生成唯一ID
        self.path = path
        self.name = name
        self.ext = ext
        self.type = ftype  # "document" / "image" / "audio" / "video" / "other"

        # 元信息
        self.metadata = {
            "size": size,
            "created_at": created_at or datetime.now().isoformat(),
            "modified_at": modified_at or datetime.now().isoformat()
        }

        # 内容
        self.content = {
            "text_summary": text_summary,
            "image_features": image_features,
            "audio_transcript": audio_transcript,
            "video_keyframes": video_keyframes
        }

        # Embedding & 分类信息
        self.embedding = None
        self.candidates = []
        self.final_label = None
        self.feedback = None

    # 转换成字典
    def to_dict(self):
        return {
            "file_id": self.file_id,
            "path": self.path,
            "name": self.name,
            "ext": self.ext,
            "type": self.type,
            "metadata": self.metadata,
            "content": self.content,
            "embedding": self.embedding,
            "candidates": self.candidates,
            "final_label": self.final_label,
            "feedback": self.feedback
        }

    # 转换成JSON
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    # 更新标签
    def update_label(self, label, feedback=False):
        if feedback:
            self.feedback = label
        else:
            self.final_label = label
      # 内容字典 { "text_summary": "...", "ocr_text": "..." }
