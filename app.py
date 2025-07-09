import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os
from datetime import datetime

# ================
# CONSTANTS & CONFIG
# ================
APP_TITLE = "🤖 Chat With Your Data"
APP_SUBTITLE = "Advanced CSV analysis with AI-powered insights"

COLOR_PALETTES = {
    "Professional Blue": ("#1F77B4", "#FF7F0E"),  # Blue + Orange
    "Executive Green": ("#2CA02C", "#D62728"),    # Green + Red
    "Vibrant Purple": ("#9467BD", "#8C564B")      # Purple + Brown
}

# ================
# SIDEBAR SETUP
# ================
# Always initialize chat history at the top
if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.sidebar:
    st.markdown(f"""
    <div style='border-bottom: 1px solid #eee; padding-bottom: 1rem; margin-bottom: 1.5rem;'>
        <h2 style='color: #333;'>Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Color theme selection
    palette = st.selectbox(
        "Color Theme",
        options=list(COLOR_PALETTES.keys()),
        index=0
    )
    primary_color, secondary_color = COLOR_PALETTES[palette]
    
    # Chart controls
    st.markdown("**Chart Options**")
    col1, col2 = st.columns(2)
    with col1:
        show_bar = st.checkbox("Bar Chart", True)
        show_line = st.checkbox("Line Chart", True)
    with col2:
        show_pie = st.checkbox("Pie Chart", False)
        show_heat = st.checkbox("Heatmap", True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander('💬 Chat History', expanded=True):
        if st.session_state['history']:
            for q, a in st.session_state['history']:
                st.markdown(f"<div style='margin-bottom:0.5em;'><b style='color:#555'>You:</b> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-bottom:1em;'><b style='color:{COLOR_PALETTES[palette][0]}'>AI:</b> {a}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='margin: 1rem 0; color: #888; text-align:center;'>No messages yet. Start the conversation!</div>",
                unsafe_allow_html=True
            )

    # Chat history in sidebar (always visible if exists)
    if 'history' in st.session_state and st.session_state['history']:
        with st.expander('💬 Chat History', expanded=True):
            for q, a in st.session_state['history'][-10:]:
                st.markdown(f"<div style='margin-bottom:0.5em;'><b style='color:#555'>You:</b> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-bottom:1em;'><b style='color:{primary_color}'>AI:</b> {a}</div>", unsafe_allow_html=True)

# ================
# LLM INTEGRATION
# ================
def get_llm_pipeline():
    """Professional LLM backend selector"""
    try:
        from ollama import Client as OllamaClient
        ollama_client = OllamaClient()
        models_response = ollama_client.list()
        models = models_response.get('models', [])
        if models and isinstance(models[0], dict) and models[0].get('name'):
            model_name = models[0]['name']
        else:
            model_name = 'mistral'

        def ollama_query(prompt, history=None):
            nonlocal model_name
            if not model_name or not isinstance(model_name, str):
                model_name = 'mistral'
            messages = []
            if history:
                for h in history:
                    messages.append({"role": "user", "content": h[0]})
                    messages.append({"role": "assistant", "content": h[1]})
            messages.append({"role": "user", "content": prompt})
            with st.spinner(f"Analyzing with {model_name}..."):
                response = ollama_client.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": 0.7}
                )
            return response['message']['content']

        return ollama_query, f"Ollama ({model_name})"
    
    except Exception as e:
        st.warning(f"Ollama connection failed: {str(e)}")
        try:
            from transformers import pipeline
            nlp = pipeline("text2text-generation", model="google/flan-t5-base")
            return (lambda p, h: nlp(p)[0]['generated_text']), "Hugging Face"
        except Exception:
            return (lambda p, h: "LLM unavailable"), "None"

# ================
# DATA HANDLING
# ================
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        if df.shape[0] > 10000:
            return None, "File too large (max 10,000 rows)"
        return df, None
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            if df.shape[0] > 10000:
                return None, "File too large (max 10,000 rows)"
            return df, None
        except Exception as e2:
            try:
                df = pd.read_csv(uploaded_file, encoding='cp1252')
                if df.shape[0] > 10000:
                    return None, "File too large (max 10,000 rows)"
                return df, None
            except Exception as e3:
                return None, f"Failed to decode CSV: {str(e3)}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ================
# VISUALIZATIONS
# ================
def auto_visualizations(df):
    """Generate professional visualizations"""
    df = df.reset_index()  # Always flatten index for plotly
    figs = []
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Bar Chart
    if show_bar and len(num_cols) > 0:
        fig = px.bar(
            df.nlargest(10, num_cols[0]),
            y=num_cols[0],
            color_discrete_sequence=[primary_color],
            opacity=0.95
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(gridcolor='rgba(0,0,0,0.08)'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        figs.append(("Top Values", fig))
    
    # Pie Chart (Sidebar)
    if 'show_pie' in globals() and show_pie and len(cat_cols) > 0:
        pie_col = cat_cols[0]
        pie_data = df[pie_col].value_counts().reset_index()
        pie_data.columns = [pie_col, "Count"]
        fig = px.pie(pie_data, names=pie_col, values="Count", title=f"Pie chart of {pie_col}")
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        figs.append((f"Pie Chart: {pie_col}", fig))
    
    # Line Chart
    if show_line and len(num_cols) > 1:
        fig = px.line(
            df,
            y=num_cols[:2],
            color_discrete_sequence=[primary_color, secondary_color],
            line_shape='spline',
            markers=True
        )
        fig.update_traces(line_width=2.5, marker=dict(size=8))
        fig.update_layout(plot_bgcolor='white')
        figs.append(("Trend Analysis", fig))
    
    # Professional Correlation Heatmap
    if show_heat and len(num_cols) > 1:
        import plotly.graph_objects as go
        corr = df[num_cols].corr().round(2)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation", thickness=18),
                hoverongaps=False
            )
        )
        # Add value annotations
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                fig.add_annotation(
                    x=corr.columns[j],
                    y=corr.index[i],
                    text=str(val),
                    showarrow=False,
                    font=dict(color="white" if abs(val) > 0.5 else "black", size=12)
                )
        fig.update_layout(
            title="<b>Correlation Heatmap</b>",
            title_font_size=18,
            font=dict(size=12),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            plot_bgcolor="white",
            autosize=True,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        figs.append(("Correlations", fig))
    
    return figs

# ================
# MAIN APP
# ================
def extract_columns_from_query(query, columns):
    import re
    from difflib import get_close_matches
    words = re.findall(r'\w+', query.lower())
    matches = []
    for col in columns:
        col_clean = col.lower().replace('_', ' ').replace('-', ' ')
        for word in words:
            if word in col_clean or col_clean in word:
                matches.append(col)
                break
    # Also try close matches
    for word in words:
        close = get_close_matches(word, columns, n=1, cutoff=0.8)
        if close:
            matches.append(close[0])
    # Remove duplicates
    return list(dict.fromkeys(matches))

def main():
    st.set_page_config(layout="wide")
    st.title(APP_TITLE)
    st.markdown(f"<div style='color: #666; margin-bottom: 2rem;'>{APP_SUBTITLE}</div>", 
               unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to start chatting with your data.")
        # Optionally, render a disabled chat form for UI consistency
        with st.form(key="chat_form_disabled"):
            st.text_input("Your question:", key="user_query", disabled=True)
            st.form_submit_button("Send", disabled=True)
        return
    
    # Data loading
    df, error = load_csv(uploaded_file)
    if error or df is None:
        st.error(error or "Failed to load CSV.")
        # Optionally, render a disabled chat form for UI consistency
        with st.form(key="chat_form_disabled"):
            st.text_input("Your question:", key="user_query", disabled=True)
            st.form_submit_button("Send", disabled=True)
        return
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head(3), use_container_width=True)
    
    # Visualizations
    with st.expander("📊 Visualizations", expanded=True):
        figs = auto_visualizations(df)
        for name, fig in figs:
            st.markdown(f"**{name}**")
            st.plotly_chart(fig, use_container_width=True)
    
    # Chat interface
    with st.expander("💬 Ask About Your Data", expanded=True):
        llm_query, llm_backend = get_llm_pipeline()
        st.info(f"Using: {llm_backend}")
        
        if 'history' not in st.session_state:
            st.session_state['history'] = []
            
        import re
        def clean_html(s):
            return re.sub(r'</?div[^>]*>', '', s)
        # First show chat history
        with st.container():
            if st.session_state['history']:
                for idx, (q, a) in enumerate(st.session_state['history']):
                    if q == "[Plot]":
                        # Only show the chart (no bubble)
                        st.plotly_chart(a, use_container_width=True, key=f"plotly_chart_{idx}")
                    else:
                        # User bubble (right)
                        st.markdown(
                            f"""
                            <div style='display: flex; justify-content: flex-end; margin: 0.5rem 0;'>
                                <div style='background: #f5f7fa; color: #333; padding: 0.75rem 1rem; border-radius: 16px 16px 0 16px; max-width: 70%; text-align: right;'>
                                    <b>You:</b> {q}
                                </div>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        # AI bubble (left)
                        a_clean = clean_html(a)
                        st.markdown(
                            f"""
                            <div style='display: flex; justify-content: flex-start; margin: 0.2rem 0 1.2rem 0;'>
                                <div style='background: {primary_color}; color: white; padding: 0.75rem 1rem; border-radius: 16px 16px 16px 0; max-width: 70%; text-align: left;'>
                                    <b>AI:</b> {a_clean}
                                </div>
                            </div>
                            """, unsafe_allow_html=True
                        )
            else:
                st.markdown(
                    "<div style='margin: 1rem 0; color: #888; text-align:center;'>No messages yet. Start the conversation!</div>",
                    unsafe_allow_html=True
                )
        # Then show the chat input at the bottom
        # Use st.form for chat input: pressing Enter submits
        # Chat input form (always at the bottom)
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input(
                "Your question:",
                key="user_query"
            )
            submitted = st.form_submit_button("Send")

        # Helper to detect chart requests
        def detect_chart_request(query):
            q = query.lower()
            if "scatter" in q:
                return "scatter"
            if "line" in q:
                return "line"
            if "bar" in q:
                return "bar"
            if "pie" in q:
                return "pie"
            if "heatmap" in q or "correlation" in q:
                return "heatmap"
            return None

        # Process the input IMMEDIATELY after form submission
        if submitted and user_query:
            # --- Data summary prompt improvement ---
            summary_keywords = ["summary", "summarize", "overview", "whole data", "all data", "entire dataset", "describe"]
            if any(k in user_query.lower() for k in summary_keywords):
                # Build a concise summary string
                summary_lines = []
                summary_lines.append(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                # Date range if present
                date_col = None
                for c in df.columns:
                    if re.search(r'date|time|est', c, re.I):
                        date_col = c
                        break
                if date_col is not None:
                    try:
                        dates = pd.to_datetime(df[date_col], errors='coerce')
                        min_date = dates.min()
                        max_date = dates.max()
                        if pd.notnull(min_date) and pd.notnull(max_date):
                            summary_lines.append(f"{date_col} range: {min_date.date()} to {max_date.date()}")
                    except Exception:
                        pass
                # Show numeric stats for up to 3 main numeric columns (excluding date)
                num_cols = [c for c in df.select_dtypes(include='number').columns if c != date_col][:3]
                for c in num_cols:
                    stats = df[c].agg(['mean','std','min','max','count']).to_dict()
                    summary_lines.append(f"{c}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']}, max={stats['max']}, count={int(stats['count'])}")
                # Show up to 2 categorical columns with a sample of unique values
                cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c != date_col][:2]
                for c in cat_cols:
                    uniques = df[c].dropna().unique()[:3]
                    summary_lines.append(f"{c}: sample values={list(uniques)}")
                summary_txt = "\n".join(summary_lines)
                prompt = f"DATA SUMMARY:\n{summary_txt}\n\nUSER QUESTION: {user_query}\n\n---\nPlease organize your answer as a well-formatted Markdown summary with clear sections and bullet points or tables. Example format:\n\n**Dataset Overview**\n- Rows: ...\n- Columns: ...\n- Date range: ...\n\n**Numeric Columns**\n| Column | Mean | Std | Min | Max | Count |\n|--------|------|-----|-----|-----|-------|\n| ...    | ...  | ... | ... | ... | ...   |\n\n**Categorical Columns**\n- Column: sample values: ...\n- ...\n"
                st.session_state['history'].append((user_query, "Analyzing..."))
                answer = llm_query(prompt, history=[(q, a) for q, a in st.session_state['history'] if q != '[Plot]'])
                st.session_state['history'][-1] = (user_query, answer)
                st.rerun()
            chart_type = detect_chart_request(user_query)
            if chart_type:
                import re
                from difflib import get_close_matches
                num_cols = df.select_dtypes(include='number').columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                # --- Pie Chart Logic ---
                if chart_type == "pie":
                    from collections import Counter
                    # Use extract_columns_from_query to find requested categorical column
                    # Search all columns for a match, not just categorical
                    requested_cols = extract_columns_from_query(user_query, df.columns)
                    pie_col = None
                    st.write(f"[DEBUG] User query: {user_query}")
                    st.write(f"[DEBUG] DataFrame columns: {list(df.columns)}")
                    st.write(f"[DEBUG] Extracted pie column candidates: {requested_cols}")
                    if requested_cols:
                        pie_col = requested_cols[0]
                        if pie_col not in df.columns:
                            st.warning(f"Requested column '{pie_col}' not found in the data. Available columns: {list(df.columns)}.")
                            answer = f"Column '{pie_col}' does not exist."
                            st.session_state['history'].append((user_query, answer))
                            st.rerun()
                    elif cat_cols:
                        # Avoid EST/date columns unless explicitly requested
                        non_date_cats = [c for c in cat_cols if not re.search(r'est|date|time', c, re.I)]
                        pie_col = non_date_cats[0] if non_date_cats else cat_cols[0]
                    if pie_col is not None:
                        if pie_col.lower() == 'events':
                            # Special handling for events: split multi-events, treat empty/NaN as 'None'
                            events_series = df[pie_col].fillna('').replace('', 'None')
                            all_events = []
                            for cell in events_series:
                                for event in str(cell).split('-'):
                                    event = event.strip()
                                    if not event or event.lower() == 'nan':
                                        event = 'None'
                                    all_events.append(event)
                            pie_data = pd.Series(all_events).value_counts().reset_index()
                            pie_data.columns = [pie_col, "Count"]
                        else:
                            pie_data = df[pie_col].fillna('None').replace('', 'None').value_counts().reset_index()
                            pie_data.columns = [pie_col, "Count"]
                        with st.spinner('Rendering pie chart...'):
                            fig = px.pie(pie_data, names=pie_col, values="Count", title=f"Pie chart of {pie_col}")
                        st.session_state['history'].append((user_query, "[Pie Chart]"))
                        st.session_state['history'].append(("[Plot]", fig))
                        answer = None
                    else:
                        answer = "No suitable categorical column found for pie chart."
                    # Skip rest of chart logic
                    st.rerun()
                fig = None
                chart_msg = None
                requested_cols = extract_columns_from_query(user_query, df.columns)
                # For scatter/line/bar, need two numeric columns
                def pick_two_numeric(cols):
                    found = [col for col in cols if col in num_cols]
                    if len(found) >= 2:
                        return found[:2]
                    if len(found) == 1 and len(num_cols) > 1:
                        others = [col for col in num_cols if col != found[0]]
                        return [found[0], others[0]] if others else None
                    if len(num_cols) >= 2:
                        return num_cols[:2]
                    return None
                if chart_type in ["scatter", "line", "bar"]:
                    chosen = pick_two_numeric(requested_cols)
                    if chosen:
                        x, y = chosen
                        if chart_type == "scatter":
                            fig = px.scatter(df, x=x, y=y, color_discrete_sequence=[primary_color])
                            chart_msg = f"Scatter plot of {x} vs {y}"
                        elif chart_type == "line":
                            fig = px.line(df, x=x, y=y, color_discrete_sequence=[primary_color])
                            chart_msg = f"Line chart of {y} over {x}"
                        elif chart_type == "bar":
                            fig = px.bar(df, x=x, y=y, color_discrete_sequence=[primary_color])
                            chart_msg = f"Bar chart of {y} by {x}"
                    else:
                        chart_msg = "Could not find two numeric columns for this chart."
                elif chart_type == "pie" and len(cat_cols) >= 1 and len(num_cols) >= 1:
                    fig = px.pie(df, names=cat_cols[0], values=num_cols[0], color_discrete_sequence=[primary_color, secondary_color])
                    chart_msg = f"Pie chart of {cat_cols[0]} by {num_cols[0]}"
                elif chart_type == "heatmap" and len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        aspect="auto",
                        title="Correlation Heatmap"
                    )
                    chart_msg = "Correlation heatmap of numeric columns"
                else:
                    chart_msg = "Not enough columns for this chart type."
                st.session_state['history'].append((user_query, chart_msg))
                if fig is not None:
                    st.session_state['history'].append(("[Plot]", fig))
                st.rerun()
            else:
                # 1. Instantly show user message and placeholder AI response
                st.session_state['history'].append((user_query, "Analyzing..."))
                st.rerun()

        # 2. After rerun, if the last AI answer is 'Analyzing...', compute the real answer
        if st.session_state['history'] and st.session_state['history'][-1][1] == "Analyzing...":
            user_query = st.session_state['history'][-1][0]
            def try_direct_answer(query, df):
                import re
                import streamlit as st
                q = query.lower()
                q_clean = re.sub(r'[^\w\s]', '', q)

                # --- Data Quality: Missing/null/empty ---
                if any(k in q_clean for k in ["missing", "null", "empty", "nan"]):
                    missing = df.isnull().sum().to_dict()
                    code = 'missing_values = ' + str(missing)
                    return f"```python\n{code}\n```"
                # --- Data Quality: Unique values ---
                if ("unique" in q_clean and ("value" in q_clean or "values" in q_clean)) or re.search(r"how many unique|number of unique|count of unique", q_clean):
                    # Try to find the target column
                    words = q_clean.split()
                    for col in df.columns:
                        col_clean = col.lower().replace('_', ' ').replace('-', ' ')
                        if col_clean in q_clean or any(word in col_clean for word in words) or any(word in col_clean for word in words):
                            nuniq = df[col].nunique()
                            return f"```python\nunique_{col.lower().replace(' ', '_')} = {nuniq}\n```"
                    # If not found, show all
                    uniques = {col: df[col].nunique() for col in df.columns}
                    code = 'unique_values = ' + str(uniques)
                    return f"```python\n{code}\n```"
                # --- Data Quality: Variance ---
                if "variance" in q_clean or "variability" in q_clean:
                    v = df.var(numeric_only=True).sort_values(ascending=False)
                    code = 'variances = ' + v.to_dict().__repr__()
                    return f"```python\n{code}\n```"
                # --- Numeric/statistics ---
                # --- Hardened Numeric/statistics ---
                op_map = [
                    ("mean", ["average", "mean"], lambda x: x.mean()),
                    ("sum", ["sum", "total", "sum of"], lambda x: x.sum()),
                    ("min", ["minimum", "lowest", "smallest", "min"], lambda x: x.min()),
                    ("max", ["maximum", "highest", "largest", "max"], lambda x: x.max()),
                    ("count", ["count", "how many", "number of", "count of"], lambda x: x.count()),
                ]
                found_keyword = None
                for op, keywords, func in op_map:
                    if any(k in q_clean for k in keywords):
                        found_keyword = (op, func)
                        break
                st.write(f"[DEBUG] Detected op: {found_keyword}")
                if not found_keyword:
                    return None  # No numeric keyword, skip direct answer

                # Robust fuzzy/partial column matching
                col_matches = []
                words = q_clean.split()
                for col in df.columns:
                    col_clean = col.lower().replace("_", " ").replace("-", " ")
                    if col_clean in q_clean or any(word in col_clean for word in words) or any(word in q_clean.split() for word in col_clean.split()):
                        col_matches.append(col)
                st.write(f"[DEBUG] Column matches: {col_matches}")
                # Prefer numeric columns
                num_col_matches = [col for col in col_matches if pd.api.types.is_numeric_dtype(df[col])]
                st.write(f"[DEBUG] Numeric column matches: {num_col_matches}")
                if len(num_col_matches) == 1:
                    col = num_col_matches[0]
                    op, func = found_keyword
                    col_data = df[col]
                    val = func(col_data)
                    var_name = f"{op}_{col.lower().replace(' ', '_').replace('-', '_')}"
                    if isinstance(val, (int, float)):
                        return f"```python\n{var_name} = {val:.2f}\n```"
                    else:
                        return f"```python\n{var_name} = {val}\n```"
                # If only one match (even if not numeric), try to convert to numeric
                if len(col_matches) == 1:
                    col = col_matches[0]
                    op, func = found_keyword
                    col_data = pd.to_numeric(df[col], errors="coerce")
                    col_data = col_data.dropna()
                    if len(col_data) == 0:
                        return None
                    val = func(col_data)
                    var_name = f"{op}_{col.lower().replace(' ', '_').replace('-', '_')}"
                    if isinstance(val, (int, float)):
                        return f"```python\n{var_name} = {val:.2f}\n```"
                    else:
                        return f"```python\n{var_name} = {val}\n```"
                # If only one numeric column in df, use it
                num_cols = df.select_dtypes(include='number').columns
                if len(num_cols) == 1:
                    col = num_cols[0]
                    op, func = found_keyword
                    col_data = df[col]
                    val = func(col_data)
                    var_name = f"{op}_{col.lower().replace(' ', '_').replace('-', '_')}"
                    if isinstance(val, (int, float)):
                        return f"```python\n{var_name} = {val:.2f}\n```"
                    else:
                        return f"```python\n{var_name} = {val}\n```"
                # If multiple matches, but only one is numeric
                if len(num_col_matches) == 1:
                    col = num_col_matches[0]
                    op, func = found_keyword
                    col_data = df[col]
                    val = func(col_data)
                    var_name = f"{op}_{col.lower().replace(' ', '_').replace('-', '_')}"
                    if isinstance(val, (int, float)):
                        return f"```python\n{var_name} = {val:.2f}\n```"
                    else:
                        return f"```python\n{var_name} = {val}\n```"
                st.write("[DEBUG] No confident match for direct answer.")
                return None  # No confident match
            answer = try_direct_answer(user_query, df)
            if answer is None:
                data_info = f"Data columns: {', '.join(df.columns)}. First row: {df.iloc[0].to_dict()}"
                prompt = f"You are a helpful data analyst AI. Here is info about the data: {data_info}. Now answer this question: {user_query}"
                answer = llm_query(prompt)
            st.session_state['history'][-1] = (user_query, answer)
            st.rerun()

if __name__ == "__main__":
    main()