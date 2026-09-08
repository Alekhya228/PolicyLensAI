import os
import pandas as pd
from src.pipeline import PolicyLensPipeline
from src.generation.llm import OllamaGenerator

class RAGEvaluator:
    def __init__(self, model_name="phi3"):
        self.llm = OllamaGenerator(model_name=model_name)
        
    def score_faithfulness(self, question, context, answer):
        """Uses LLM-as-a-judge to check if the answer is grounded in the context."""
        prompt = f"""You are an impartial judge evaluating a RAG system.
Given the QUESTION, CONTEXT, and ANSWER, your job is to determine if the ANSWER contains any hallucinations or unverified claims not found in the CONTEXT.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Respond with exactly one word: 'YES' if the answer is completely faithful to the context, or 'NO' if it contains hallucinations or outside information."""
        
        response = self.llm.generate(prompt).strip().upper()
        return 1.0 if "YES" in response else 0.0

    def score_relevance(self, question, answer):
        """Uses LLM-as-a-judge to check if the answer actually addresses the question."""
        prompt = f"""You are an impartial judge.
Given the QUESTION and the ANSWER, determine if the ANSWER successfully and directly addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Respond with exactly one word: 'YES' if the answer is relevant and addresses the question, or 'NO' if it dodges the question or is irrelevant."""

        response = self.llm.generate(prompt).strip().upper()
        return 1.0 if "YES" in response else 0.0

def run_evaluation():
    print("Starting LLM-as-a-Judge Evaluation Pipeline...\n")
    
    test_questions = [
        "What are the GPA points for a C grade?",
        "What happens if a student is caught cheating on an exam?",
        "Can I get an incomplete grade if I finished 50% of the coursework?"
    ]
    
    pipeline = PolicyLensPipeline(model_name="phi3")
    pipeline.ingest_documents()
    evaluator = RAGEvaluator(model_name="phi3")
    
    results = []
    
    print("\nEvaluating queries...")
    for q in test_questions:
        print(f"\nQuery: {q}")
        res = pipeline.query(q, top_k=3)
        answer = res['answer']
        
        # Combine chunks into a single context string
        context_str = "\n".join([chunk['content'] for chunk in res['chunks_used']])
        
        # Score
        faith_score = evaluator.score_faithfulness(q, context_str, answer)
        rel_score = evaluator.score_relevance(q, answer)
        
        print(f"Faithfulness Score: {faith_score}")
        print(f"Relevance Score:    {rel_score}")
        
        results.append({
            "question": q,
            "faithfulness": faith_score,
            "relevance": rel_score
        })
        
    df = pd.DataFrame(results)
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(df)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/evaluation_metrics.csv", index=False)
    print("\nDetailed metrics saved to data/evaluation_metrics.csv")

if __name__ == "__main__":
    run_evaluation()
