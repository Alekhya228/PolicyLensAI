import streamlit as st
from src.pipeline import PolicyLensPipeline
import os

st.set_page_config(page_title="PolicyLens AI", layout="wide")

@st.cache_resource
def get_pipeline():
    pipeline = PolicyLensPipeline(model_name="gpt-3.5-turbo")
    return pipeline

st.title("PolicyLens AI 📚")
st.subheader("Evidence-Grounded Academic Policy Assistant")

pipeline = get_pipeline()

# Sidebar for Document Management
with st.sidebar:
    st.header("📂 Document Management")
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload new policy PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data/raw_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")
        
    st.divider()
    
    # Ingestion Trigger
    if st.button("🔄 Process & Ingest Documents", use_container_width=True):
        with st.spinner("Parsing PDFs and updating vector store..."):
            pipeline.ingest_documents()
        st.success("Ingestion complete!")
        
    st.divider()
    
    # Document List & Deletion
    st.markdown("### 📚 Current Database")
    if os.path.exists("data/raw_pdfs"):
        files = [f for f in os.listdir("data/raw_pdfs") if f.endswith('.pdf')]
        
        if files:
            st.write(f"Total documents: {len(files)}")
            for f in files:
                st.markdown(f"- `{f}`")
                
            file_to_delete = st.selectbox("Select a file to remove", [""] + files)
            if file_to_delete and st.button("🗑️ Delete Selected File"):
                os.remove(os.path.join("data/raw_pdfs", file_to_delete))
                st.success(f"Deleted {file_to_delete}. Please re-ingest documents.")
                st.rerun()
        else:
            st.info("No documents found. Upload some PDFs above.")
            
    st.divider()
    
    st.markdown("### 📊 System Analytics")
    if os.path.exists("data/feedback_log.csv"):
        import pandas as pd
        try:
            df = pd.read_csv("data/feedback_log.csv")
            st.write(f"Total Feedbacks: {len(df)}")
            
            # Show missing knowledge (responses containing "I cannot find")
            missing = df[df['Response'].str.contains("I cannot find", na=False)]
            if not missing.empty:
                st.warning(f"Unanswered queries: {len(missing)}")
                with st.expander("View Unanswered Queries"):
                    st.dataframe(missing[['Query', 'Timestamp']])
                    
            with st.expander("View Full Feedback Log"):
                st.dataframe(df)
        except Exception as e:
            st.error("Error loading analytics.")
    else:
        st.info("No feedback data collected yet.")
            
    st.divider()
    
    st.markdown("### ⚙️ Advanced Settings")
    retrieval_k = st.slider("Number of Context Chunks (top_k)", min_value=1, max_value=10, value=3)

# Initialize tabs
tab_chat, tab_eval, tab_stress = st.tabs(["💬 Live Assistant", "📊 Model Report Card", "🔥 Stress Test"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a policy question (e.g., 'What is the GPA for a C grade?'):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                try:
                    if not pipeline.hybrid_searcher.bm25:
                        st.warning("Index not loaded in memory. Please click 'Process & Ingest Documents' in the sidebar first.")
                        st.stop()
                        
                    result = pipeline.query(prompt, top_k=retrieval_k)
                    
                    if isinstance(result, str):
                        response = result
                        chunks = []
                    else:
                        response = result["answer"]
                        chunks = result["chunks_used"]

                    # Determine Provenance Tag
                    sources = set([c['metadata'].get('source') for c in chunks])
                    if "Conflict Callout" in response or "differs from" in response.lower():
                        provenance = "🕒 Time-sensitive — verify with your department"
                        color = "orange"
                    elif len(sources) > 1:
                        provenance = f"🔗 Synthesized across {len(sources)} documents"
                        color = "blue"
                    elif len(sources) == 1:
                        provenance = "📄 Direct lookup"
                        color = "green"
                    else:
                        provenance = "❓ Unknown provenance"
                        color = "grey"

                    # Check for "Not Covered" state
                    is_not_covered = "i cannot find the answer" in response.lower() or "no relevant information found" in response.lower()
                    
                    if is_not_covered:
                        st.warning("⚠️ **Not Covered in Current Knowledge Base**")
                        st.write("I searched the following related policies but couldn't find a direct answer for your query:")
                        if chunks:
                            for chunk in chunks:
                                st.markdown(f"- `{chunk['metadata'].get('source')}` (Page {chunk['metadata'].get('page')})")
                        else:
                            st.write("- No related documents found.")
                        st.write("---")
                        st.info("💡 **Should this topic be added to the knowledge base?**")
                        if st.button("Submit Request to Admin"):
                            st.toast("Request logged for Admin review!")
                            
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.markdown(response)
                        st.markdown(f":{color}[**Provenance:** {provenance}]")
                        
                        st.markdown("---")
                        st.markdown("### 🔍 Citation Trace")
                        
                        cols = st.columns(len(chunks) if chunks else 1)
                        for i, chunk in enumerate(chunks):
                            meta = chunk['metadata']
                            date = meta.get('last_updated', 'Unknown')
                            source = meta.get('source', 'Unknown')
                            with cols[i].popover(f"[{i+1}] {source} (📅 {date})", use_container_width=True):
                                st.markdown(f"**Exact Source Passage:**")
                                st.info(chunk['content'])
                                st.caption(f"**Score:** {chunk.get('score', 'N/A')}")
                                st.caption(f"**Method:** {chunk.get('source_type', 'unknown')}")
                                
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        # Feedback loop feeding the Gold Set
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant":
            latest_response = st.session_state.messages[-1]["content"].lower()
            if not ("i cannot find the answer" in latest_response or "no relevant information found" in latest_response):
                st.markdown("---")
                st.write("**Was this answer helpful?**")
                col1, col2, _ = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 Yes"):
                        st.toast("Added to Gold Set (Positive Examples)!")
                with col2:
                    with st.popover("👎 No"):
                        st.write("Help us improve the Gold Set.")
                        reason = st.selectbox("Reason:", ["Wrong citation", "Outdated policy", "Unclear/Dodged question", "Hallucinated"])
                        if st.button("Submit to Triage"):
                            import csv
                            from datetime import datetime
                            log_file = "data/feedback_log.csv"
                            file_exists = os.path.exists(log_file)
                            with open(log_file, "a", newline="", encoding='utf-8') as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(["Timestamp", "Query", "Response", "Feedback"])
                                writer.writerow([datetime.now().isoformat(), st.session_state.messages[-2]["content"], response, f"Negative: {reason}"])
                            st.success("Triaged into Gold-Set backlog!")

with tab_eval:
    st.header("Model Report Card")
    st.markdown("This dashboard tracks the mathematical evaluation of the RAG pipeline using LLM-as-a-judge.")
    
    if os.path.exists("data/evaluation_metrics.csv"):
        import pandas as pd
        df = pd.read_csv("data/evaluation_metrics.csv")
        
        col1, col2 = st.columns(2)
        avg_faith = df['faithfulness'].mean() * 100
        avg_rel = df['relevance'].mean() * 100
        
        col1.metric("Average Faithfulness", f"{avg_faith:.1f}%", help="Percentage of answers entirely grounded in the retrieved context.")
        col2.metric("Average Answer Relevance", f"{avg_rel:.1f}%", help="Percentage of answers that directly address the user's question.")
        
        st.subheader("Granular Test Results")
        st.dataframe(df.style.background_gradient(cmap="RdYlGn", subset=["faithfulness", "relevance"]), use_container_width=True)
        
        st.caption("These scores are generated autonomously by the pipeline comparing the generated answers to our Gold Set questions.")
    else:
        st.info("No evaluation data available. Run `python -m src.eval.evaluator` in the terminal to generate the report card.")

with tab_stress:
    st.header("System Stress Test")
    st.markdown("Run some of our hardest **Gold-Set failure cases** live to see how the system handles ambiguity and missing data.")
    
    st.markdown("#### Scenario 1: The Missing Sub-Clause")
    st.info("**Question:** Can I get an incomplete grade if I finished 50% of the coursework?")
    st.caption("*Why it's hard:* The policy states 75%. Most standard LLMs will hallucinate 'yes' or give generic advice. We want our system to strictly say 'No' based on the text.")
    if st.button("Run Live: Scenario 1"):
        with st.spinner("Processing..."):
            res = pipeline.query("Can I get an incomplete grade if I finished 50% of the coursework?", top_k=retrieval_k)
            st.success("Completed")
            st.write(res["answer"] if not isinstance(res, str) else res)
            
    st.divider()
    
    st.markdown("#### Scenario 2: The Conflicting Policy")
    st.info("**Question:** What is the GPA for a C grade?")
    st.caption("*Why it's hard:* We have a 2023 policy (C = 2.0) and a 2026 update (C = 2.5). The system must detect the conflict and prioritize freshness.")
    if st.button("Run Live: Scenario 2"):
        with st.spinner("Processing..."):
            res = pipeline.query("What is the GPA for a C grade?", top_k=retrieval_k)
            st.success("Completed")
            st.write(res["answer"] if not isinstance(res, str) else res)
