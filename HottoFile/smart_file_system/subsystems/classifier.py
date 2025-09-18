import subprocess
import json

class OllamaClassifier:
    def __init__(self, model="gemma3"):
        self.model = model

    def classify(self, file_info, candidate_labels):
        text_summary = file_info.content.get('text_summary', '')
        if not text_summary:
            return None

        prompt = f"""
你是一个文档分类助手。请根据下面的文档摘要，在候选类别中选择最合适的一个类别，并只输出类别名称。

文档摘要:
{text_summary}

候选类别:
{', '.join(candidate_labels)}

请只输出一个类别名称，且必须严格来自候选类别。
        """

        try:
            # 调用 ollama run
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                text=True
            )
            output = result.stdout.strip()

            # 去掉换行符、多余字符
            output = output.split("\n")[0].strip()
            return output if output in candidate_labels else None
        except Exception as e:
            print(f"[Error] Ollama classify failed: {e}")
            return None
class Classifier:
    def __init__(self, llm_classifier: OllamaClassifier):
        self.llm_classifier = llm_classifier

    def _classify_by_extension(self, file_ext):
        ext_map = {
            'pdf': 'document',
            'docx': 'document',
            'pptx': 'document',
            'txt': 'document',
            'md': 'document',
            'xlsx': 'document',
            'csv': 'document',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'gif': 'image',
            'py': 'code',
            'js': 'code',
            'html': 'code',
            'css': 'code',
        }
        return ext_map.get(file_ext.lower(), 'other')

    def classify(self, file_info, embedding_manager):
        # 1. Rule-based 分类
        rule_based_type = self._classify_by_extension(file_info.ext)
        file_info.type = rule_based_type

        if file_info.content.get('text_summary'):
            # 2. Embedding-based 生成候选类别
            search_results = embedding_manager.search(file_info.content['text_summary'], k=5)
            candidate_labels = list(set(
                [res['file'].final_label for res in search_results if res['file'].final_label]
            ))

            if not candidate_labels:
                candidate_labels = ["research paper", "code script", "contract", "invoice", "image", "other"]

            file_info.candidates = candidate_labels

            # 3. LLM-based 分类
            if self.llm_classifier:
                final_label = self.llm_classifier.classify(file_info, candidate_labels)
                if final_label is None:
                    final_label = rule_based_type  # fallback
                file_info.final_label = final_label
            else:
                file_info.final_label = rule_based_type
        else:
            file_info.final_label = rule_based_type

        return file_info