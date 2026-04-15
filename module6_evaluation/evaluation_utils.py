# evaluation_utils.py
"""
Module 6: Evaluation & Analysis - Core Utilities
Reusable functions for RAG evaluation.
"""

import re
import time
from typing import List, Dict, Any
import pandas as pd
import requests


class OllamaChat:
    """Ollama LLM 客户端"""
    def __init__(self, model="gemma3:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 100) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        response = requests.post(url, json=payload)
        return response.json().get("response", "").strip()


class LLMJudge:
    """LLM-as-a-judge 评估器"""
    
    def __init__(self, model_name: str = "gemma3:4b"):
        self.llm = OllamaChat(model=model_name)
    
    def evaluate_relevance(self, query: str, answer: str) -> int:
        prompt = f"""Rate the relevance of the answer (1-5). Question: {query}\nAnswer: {answer}\nReturn ONLY a number."""
        response = self.llm.generate(prompt)
        try:
            return max(1, min(5, int(re.search(r'\d+', response).group())))
        except:
            return 3
    
    def evaluate_correctness(self, query: str, answer: str) -> int:
        prompt = f"""Rate the factual correctness (1-5). Question: {query}\nAnswer: {answer}\nReturn ONLY a number."""
        response = self.llm.generate(prompt)
        try:
            return max(1, min(5, int(re.search(r'\d+', response).group())))
        except:
            return 3
    
    def evaluate_helpfulness(self, query: str, answer: str) -> int:
        prompt = f"""Rate how helpful the answer is (1-5). Question: {query}\nAnswer: {answer}\nReturn ONLY a number."""
        response = self.llm.generate(prompt)
        try:
            return max(1, min(5, int(re.search(r'\d+', response).group())))
        except:
            return 3
    
    def full_evaluation(self, query: str, answer: str) -> Dict[str, int]:
        return {
            "relevance": self.evaluate_relevance(query, answer),
            "correctness": self.evaluate_correctness(query, answer),
            "helpfulness": self.evaluate_helpfulness(query, answer),
        }


def evaluate_batch(results_df: pd.DataFrame) -> pd.DataFrame:
    """批量评估实验结果的便捷函数"""
    judge = LLMJudge()
    df = results_df.copy()
    df['relevance'] = 0
    df['correctness'] = 0
    df['helpfulness'] = 0
    df['overall'] = 0.0
    
    for idx, row in df.iterrows():
        scores = judge.full_evaluation(row['query'], row['answer'])
        df.loc[idx, 'relevance'] = scores['relevance']
        df.loc[idx, 'correctness'] = scores['correctness']
        df.loc[idx, 'helpfulness'] = scores['helpfulness']
        df.loc[idx, 'overall'] = sum(scores.values()) / 3
    
    return df