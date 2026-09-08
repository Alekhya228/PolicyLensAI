# PolicyLens AI — Evidence-Grounded Academic Policy Assistant

An evaluation-driven RAG system that answers complex college/university policy questions using official regulations and academic handbooks.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Generate mock data (or place your own PDFs in `data/raw_pdfs/`):
```bash
python mock_data_generator.py
```

3. Run the application:
```bash
streamlit run app.py
```

## Running Evaluations (LLM-as-a-Judge)

To run the custom LLM-as-a-Judge evaluation suite (measuring Faithfulness and Answer Relevance):
```bash
python -m src.eval.evaluator
```

This will generate an `evaluation_metrics.csv` that populates the **Model Report Card** tab in the live application.

## Continuous Evaluation & Gold Set Triage

This project is built around an **alive evaluation loop**:
* The UI features a feedback system on every answer.
* If a user clicks **👎 No**, they are prompted to attach a reason tag (e.g., "Outdated policy", "Hallucinated", "Wrong citation").
* These negative interactions are triaged directly into our **Gold Set backlog** via `data/feedback_log.csv`. 
* We use this backlog to author new edge-case evaluation questions, continuously expanding our test coverage and stress tests.

## Key Features
* **Citation Trace**: Inline, clickable citation chips that reveal the exact highlighted source passage.
* **Conflict Callouts**: Automated temporal detection of conflicting policies (e.g. older handbooks vs. newer updates).
* **Live Stress Testing**: A dedicated UI tab to run the system's hardest failure cases live.

*(Note: Ensure you have Ollama running locally with the `phi3` model, or update the model name in `src/pipeline.py`.)*
