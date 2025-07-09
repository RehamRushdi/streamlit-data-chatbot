# 🤖 Chat With Your Data

A Streamlit web app for AI-powered CSV analysis and instant visualization. Upload your CSV, ask questions in natural language, and get smart insights, summaries, and interactive charts—no coding required!

---

## Features

- **Conversational Data Analysis**: Ask questions about your data in plain English and get instant answers.
- **AI-Powered Summaries**: Automatic, well-formatted summaries of your dataset using LLMs (Ollama or Hugging Face models).
- **Dynamic Visualizations**: Instantly generate pie, bar, line, and heatmap charts—just by asking.
- **Sidebar Controls**: Toggle chart types, switch color themes, and view chat history.
- **Robust Data Handling**: Handles missing values, multi-label columns, and large files (up to 10,000 rows).

## Technologies Used

- **Python 3.8+**
- **Streamlit** (UI framework)
- **Pandas** (data manipulation)
- **Plotly Express** (interactive charts)
- **Ollama** (optional, for local LLM)
- **Hugging Face Transformers** (fallback LLM)

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies
It's recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Set Up Ollama
- [Install Ollama](https://ollama.com/) and run the local server for faster, private LLM responses.
- Otherwise, the app will use Hugging Face's `google/flan-t5-base` model as a fallback.

### 4. Run the App
```bash
streamlit run app.py
```

### 5. Usage
- Upload a CSV file (max 10,000 rows).
- Ask questions like:
  - "Show a pie chart of events"
  - "What is the average temperature?"
  - "Summarize the dataset"
- Toggle visualizations and color themes in the sidebar.

## Example Questions
- "Create a line chart of temperature over time."
- "How many unique cities are there?"
- "Show missing values."
- "Summarize the whole data set."

## Project Structure
```
project/
├── app.py           # Main Streamlit app
├── requirements.txt # Python dependencies
```

### 🚀 First time using Streamlit? This project is a great place to start!

---

**Enjoy chatting with your data!**
