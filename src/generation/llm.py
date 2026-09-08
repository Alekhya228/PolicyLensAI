import os
import google.generativeai as genai

class GeminiGenerator:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        
    def generate(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
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
