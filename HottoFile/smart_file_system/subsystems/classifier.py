import subprocess
from typing import List, Optional

class OllamaClassifier:
    def __init__(self, model: str = "gemma3"):
        self.model = model

    def _build_prompt(self, text_summary: str, candidate_labels: List[str], fewshot_examples: Optional[List[dict]] = None) -> str:
        """
        构造 prompt，包含 few-shot 示例（如果有）。
        fewshot_examples 格式:
            [
              {"file_id":"..", "text_summary":"..", "correct_label":".."},
              ...
            ]
        """
        prompt_parts = []
        prompt_parts.append("你是一个文档分类助手。请根据文档摘要，从候选类别中选择最合适的一个类别，并且**只输出类别名称**。")
        prompt_parts.append("")

        # few-shot 示例
        if fewshot_examples:
            prompt_parts.append("以下是一些示例（供参考）：")
            for i, ex in enumerate(fewshot_examples, start=1):
                summary = ex.get("text_summary", "")
                label = ex.get("correct_label", "")
                if summary and label:
                    prompt_parts.append(f"示例 {i} 摘要: {summary}")
                    prompt_parts.append(f"示例 {i} 正确类别: {label}")
                    prompt_parts.append("")

        # 候选类别
        prompt_parts.append("候选类别（请从中选择，严格输出其中之一）：")
        for idx, lab in enumerate(candidate_labels, start=1):
            prompt_parts.append(f"{idx}. {lab}")
        prompt_parts.append("")

        # 待分类摘要
        prompt_parts.append("待分类文档摘要：")
        prompt_parts.append(text_summary)
        prompt_parts.append("")
        prompt_parts.append("注意：最终答案必须是候选类别中的**精确文本**，不要包含编号或其他说明。无法判断时输出 'other'。")

        return "\n".join(prompt_parts)

    def classify(self, file_info, candidate_labels: List[str], fewshot_examples: Optional[List[dict]] = None) -> Optional[str]:
        """
        使用 Ollama 模型进行分类。
        """
        text_summary = file_info.content.get("text_summary", "") or ""
        if not text_summary:
            return None

        if "other" not in [c.lower() for c in candidate_labels]:
            candidate_labels = candidate_labels + ["other"]

        prompt = self._build_prompt(text_summary, candidate_labels, fewshot_examples)

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                check=False
            )
            output = result.stdout.strip()
            if not output:
                return None

            first_line = output.splitlines()[0].strip()
            low = first_line.lower()

            canonical_map = {c.lower(): c for c in candidate_labels}

            # 精确匹配
            if low in canonical_map:
                return canonical_map[low]

            # 宽松匹配：包含关系
            for cand_low, cand_orig in canonical_map.items():
                if cand_low in low:
                    return cand_orig

            return None
        except Exception as e:
            print(f"[Error] Ollama classify failed: {e}")
            return None


class Classifier:
    def __init__(self, llm_classifier: OllamaClassifier, feedback_manager=None):
        self.llm_classifier = llm_classifier
        self.feedback_manager = feedback_manager

    def _classify_by_extension(self, file_ext: str) -> str:
        ext_map = {
            'pdf': 'document', 'docx': 'document', 'pptx': 'document', 'ppt': 'document',
            'txt': 'document', 'md': 'document', 'xlsx': 'document', 'xls': 'document',
            'csv': 'document', 'jpg': 'image', 'jpeg': 'image', 'png': 'image',
            'gif': 'image', 'py': 'code', 'js': 'code', 'html': 'code', 'css': 'code',
        }
        return ext_map.get(file_ext.lower(), 'other')

    def classify(self, file_info, embedding_manager):
        # 1. rule-based
        rule_based_type = self._classify_by_extension(file_info.ext)
        file_info.type = rule_based_type

        if not file_info.content.get('text_summary'):
            file_info.final_label = rule_based_type
            return file_info

        # 2. embedding-based 候选标签
        try:
            search_results = embedding_manager.search(file_info.content['text_summary'], k=5)
        except Exception:
            search_results = []

        candidate_labels = []
        seen = set()
        for res in search_results:
            if isinstance(res, dict) and 'file' in res:
                f = res['file']
                lab = getattr(f, "final_label", None) or (f.get("final_label") if isinstance(f, dict) else None)
            elif hasattr(res, "metadata"):
                lab = res.metadata.get("final_label") or res.metadata.get("label")
            elif isinstance(res, dict):
                lab = res.get("final_label") or res.get("label")
            else:
                lab = None
            if lab and lab not in seen:
                candidate_labels.append(lab)
                seen.add(lab)

        if not candidate_labels:
            candidate_labels = ["research paper", "code script", "contract", "invoice", "image", "other"]

        if "other" not in [c.lower() for c in candidate_labels]:
            candidate_labels.append("other")

        file_info.candidates = candidate_labels

        # 3. few-shot
        fewshot_examples = None
        if self.feedback_manager and hasattr(self.feedback_manager, "get_fewshot_examples"):
            try:
                fewshot_examples = self.feedback_manager.get_fewshot_examples(k=3)
            except Exception:
                pass

        # 4. LLM-based
        final_label = None
        if self.llm_classifier:
            final_label = self.llm_classifier.classify(file_info, candidate_labels, fewshot_examples=fewshot_examples)

        if not final_label:
            final_label = rule_based_type  # fallback

        file_info.final_label = final_label
        return file_info
