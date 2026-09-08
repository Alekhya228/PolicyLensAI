import requests
import json

class OllamaGenerator:
    def __init__(self, model_name="phi3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def generate(self, prompt):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error generating response: {str(e)}"
            
    def generate_rag_answer(self, query, context_chunks):
        """Generates an answer based on context, with citations."""
        
        # Format the context
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', '?')
            date = chunk['metadata'].get('last_updated', 'Unknown')
            context_text += f"\n[{i+1}] Source: {source} | Updated: {date}\n{chunk['content']}\n"
            
        prompt = f"""You are an academic policy assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I cannot find the answer in the provided documents."

IMPORTANT INSTRUCTIONS:
1. You MUST use inline citations like [1], [2] corresponding to the context chunks. 
2. If you notice conflicting information in the context (for example, an older policy from 2023 contradicts a newer update from 2026), you MUST start your response with a conflict banner in this exact format:
> ⚠️ **Conflict Callout:** A [Year] clause differs from a [Year] update on this topic. (Explain briefly)
3. After the callout, provide the actual answer using the most recent information.

Context:
{context_text}

Question: {query}
Answer:"""

        return self.generate(prompt)
