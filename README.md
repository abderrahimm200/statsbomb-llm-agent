# AFCON StatsBomb LLM Analysis Agent

A planner-based LLM agent that turns natural language football questions into
SQL queries, match analysis, insights, and pitch visualizations.

## Features
- Natural language -> SQL -> analysis -> insights -> visualization
- Planner-based agent with retries & self-refinement
- StatsBomb-style AFCON 2023 event data (SQLite)
- Streamlit UI with full debug trace
- Optional plot reading via vision model for chart interpretation

## Example Question
> For Morocco vs South Africa, where did Morocco lose possession most often
> in the middle third, and what does it say about their build-up?

## Tech Stack
- Python
- Streamlit
- SQLite (StatsBomb-style schema)
- mplsoccer
- LLM API (DeepSeek / Gemini)

## Run locally
```bash
streamlit run app.py
```
