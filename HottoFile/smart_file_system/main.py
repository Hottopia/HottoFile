from typing import List, Dict, Any

# ===== Feedback Manager =====
class FeedbackManager:
    def __init__(self):
        self.feedback_store = []

    def add_feedback(self, input_text: str, label: str, reasoning: str):
        """保存用户反馈样本"""
        self.feedback_store.append({
            "input": input_text,
            "label": label,
            "reasoning": reasoning
        })

    def get_fewshot_examples(self, k: int = 3) -> List[Dict[str, str]]:
        """取出最近的 k 条反馈样本"""
        return self.feedback_store[-k:]


# ===== Classifier =====
class Classifier:
    def __init__(self, llm, feedback_manager: FeedbackManager):
        self.llm = llm
        self.feedback_manager = feedback_manager

    def _build_prompt(self, input_text: str) -> str:
        """构造带 few-shot 样本的 prompt"""
        examples = self.feedback_manager.get_fewshot_examples()

        # 拼接 few-shot 样例
        fewshot_text = "\n".join([
            f"示例 {i+1}：\n输入：{ex['input']}\n标签：{ex['label']}\n推理：{ex['reasoning']}"
            for i, ex in enumerate(examples)
        ])

        prompt = (
            "你是一个分类模型，请根据输入文本给出分类标签，并提供简要推理。\n\n"
            "以下是一些示例：\n"
            f"{fewshot_text if fewshot_text else '（无示例，直接分类）'}\n\n"
            f"现在请处理新的输入：\n输入：{input_text}\n标签和推理："
        )
        return prompt

    def classify(self, input_text: str) -> Dict[str, Any]:
        """调用 LLM 进行分类"""
        prompt = self._build_prompt(input_text)
        response = self.llm(prompt)  # 假设 llm(prompt) -> str

        return {
            "input": input_text,
            "output": response,
            "used_examples": self.feedback_manager.get_fewshot_examples()
        }


# ===== Example: Dummy LLM =====
class DummyLLM:
    def __call__(self, prompt: str) -> str:
        # 这里是模拟逻辑，实际换成调用 openai 或其他模型
        return f"[LLM 输出模拟]\n{prompt[-100:]}"


# ===== Usage =====
if __name__ == "__main__":
    fm = FeedbackManager()
    llm = DummyLLM()
    clf = Classifier(llm, fm)

    # 添加一些反馈样本
    fm.add_feedback("今天天气真好", "闲聊", "内容轻松，不涉及任务。")
    fm.add_feedback("帮我查一下北京天气", "任务请求", "用户明确请求信息。")
    fm.add_feedback("你觉得我该买iPhone还是华为？", "决策咨询", "涉及选择和建议。")

    # 测试分类
    res = clf.classify("明天你能帮我写作业吗？")
    print(res)
