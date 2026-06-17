import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Backend URL (loaded from environment or default local host)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Page configuration
st.set_page_config(
    page_title="AWS Customer Agreement RAG Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #FF9900 0%, #E05300 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Metric Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #FF9900;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #FF9900;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Source block styling */
    .source-container {
        border-left: 3px solid #FF9900;
        background: rgba(255, 153, 0, 0.05);
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
    .source-header {
        font-weight: 600;
        color: #FF9900;
        margin-bottom: 5px;
        display: flex;
        justify-content: space-between;
    }
    .source-body {
        font-size: 0.9rem;
        color: #d1d5db;
        line-height: 1.5;
    }
    
    /* Success/Failure labels */
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-fail {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = None

# Sidebar Ingest / Configuration Panel
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", width=100)
    st.markdown("### System Controls")
    
    # Check Backend Connection and Status
    try:
        res = requests.get(f"{BACKEND_URL}/")
        if res.status_code == 200:
            st.session_state.system_status = res.json()
            st.success("Backend API: Online")
        else:
            st.session_state.system_status = None
            st.error("Backend API: Offline")
    except Exception:
        st.session_state.system_status = None
        st.error("Backend API: Connection Failed")

    # Ingestion trigger
    st.markdown("---")
    st.markdown("#### Document Ingestion")
    
    if st.session_state.system_status:
        vdb_initialized = st.session_state.system_status.get("vector_store_initialized", False)
        if vdb_initialized:
            st.info("Vector database is initialized.")
        else:
            st.warning("Vector database is empty. Please run Ingestion.")
            
        # File uploader
        uploaded_pdf = st.file_uploader("Upload AWS Agreement PDF", type=["pdf"])
        
        if st.button("🚀 Ingest / Reload PDF", use_container_width=True):
            with st.spinner("Processing PDF, extracting sections, generating chunks and vector embeddings..."):
                try:
                    files = None
                    if uploaded_pdf is not None:
                        files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                        
                    res_ingest = requests.post(f"{BACKEND_URL}/ingest", files=files)
                    if res_ingest.status_code == 201:
                        data_ingest = res_ingest.json()
                        st.success(f"Ingested {data_ingest['chunks_created']} chunks successfully!")
                        
                        # Refresh backend status
                        res_status = requests.get(f"{BACKEND_URL}/")
                        if res_status.status_code == 200:
                            st.session_state.system_status = res_status.json()
                            st.rerun()
                    else:
                        st.error(f"Ingestion failed: {res_ingest.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to ingest endpoint: {e}")
    else:
        st.error("Cannot run ingestion. Start the backend API first.")

    st.markdown("---")
    st.markdown("#### System Settings")
    if st.session_state.system_status:
        config = st.session_state.system_status.get("config", {})
        st.markdown(f"**Embedding Model:** `{config.get('embedding_model', '').split('/')[-1]}`")
        st.markdown(f"**LLM Model:** `{config.get('llm_model', '')}`")
        st.markdown(f"**Threshold:** `{config.get('similarity_threshold', '')}`")
        st.markdown(f"**Top K Chunks:** `{config.get('top_k', '')}`")
    else:
        st.caption("No configurations loaded.")

# Header Layout
st.markdown("<div class='main-title'>AWS Customer Agreement RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Production-grade Retrieval-Augmented Generation Legal Assistant</div>", unsafe_allow_html=True)

# Main Navigation Tab Layout
tab_chat, tab_analytics = st.tabs(["💬 Chat Assistant", "📈 Analytics Dashboard"])

# TAB 1: Chat Interface
with tab_chat:
    # Check if vector DB is ready
    vdb_initialized = False
    if st.session_state.system_status:
        vdb_initialized = st.session_state.system_status.get("vector_store_initialized", False)
        
    if not vdb_initialized:
        st.warning("⚠️ The Vector Database is empty. Please upload or ingest the AWS Customer Agreement PDF using the 'System Controls' panel in the sidebar to start chatting.")
    else:
        # Layout for Chat: Left side chat, Right side detailed matching source documents
        chat_col, source_col = st.columns([2, 1.2])
        
        with chat_col:
            st.markdown("### Chat History")
            
            # Display chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        # Small expander in message
                        with st.expander("Show page citations"):
                            for idx, src in enumerate(msg["sources"]):
                                st.markdown(f"**Source {idx+1}:** Section {src['section']} | Page {src['page_num']} (Score: {src['score']:.4f})")
            
            # Text Input for question
            if prompt := st.chat_input("Ask a question about the AWS Customer Agreement..."):
                # Append user message to state
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Immediate Rerun to display user message
                st.rerun()
                
        # Handle generating response if last message is from user
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_prompt = st.session_state.messages[-1]["content"]
            
            with chat_col:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    sources_placeholder = st.empty()
                    
                    with st.spinner("Analyzing document and formulating answer..."):
                        try:
                            # Send request to FastAPI
                            res_ask = requests.post(
                                f"{BACKEND_URL}/ask",
                                json={"query": user_prompt}
                            )
                            
                            if res_ask.status_code == 200:
                                data_ask = res_ask.json()
                                response_text = data_ask["answer"]
                                sources_data = data_ask["sources"]
                                latency = data_ask["latency_ms"]
                                
                                # Render response
                                message_placeholder.markdown(response_text)
                                st.caption(f"Latency: {latency} ms")
                                
                                # Update chat history in state
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response_text,
                                    "sources": sources_data
                                })
                                
                                # Force rerun to sync sidebar state or history if needed
                                st.rerun()
                                
                            else:
                                err_detail = res_ask.json().get('detail', 'Unknown backend error')
                                message_placeholder.error(f"Error formulating response: {err_detail}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Failed to get response: {err_detail}"
                                })
                        except Exception as e:
                            message_placeholder.error(f"API Connection error: {e}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Connection error: {e}"
                            })
                            
        # Render the Sources in the right column for the last assistant response
        with source_col:
            st.markdown("### Document Citations")
            
            assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
            if assistant_messages and "sources" in assistant_messages[-1] and assistant_messages[-1]["sources"]:
                last_sources = assistant_messages[-1]["sources"]
                
                # Check if similarity failed
                answer_text = assistant_messages[-1]["content"]
                if "not present in the AWS Customer Agreement" in answer_text:
                    st.markdown("🚫 **Query classified as Out of Context**")
                    st.info("The retrieved text segments did not meet the similarity threshold. Safe answer returned without calling Gemini API.")
                
                for idx, src in enumerate(last_sources):
                    st.markdown(f"""
                    <div class='source-container'>
                        <div class='source-header'>
                            <span>📄 Source {idx+1}: {src['section']}</span>
                            <span class='badge-success'>Match: {src['score']*100:.1f}%</span>
                        </div>
                        <div class='source-body'>
                            <strong>Page:</strong> {src['page_num']}<br/>
                            <strong>Snippet:</strong> <em>\"{src['raw_text'][:250]}...\"</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Ask a question to see matching PDF document sections and citation scores in real-time.")

# TAB 2: Analytics Dashboard
with tab_analytics:
    st.markdown("### System Operations & Usage Metrics")
    
    # Reload button for analytics
    if st.button("🔄 Refresh Dashboard Metrics", use_container_width=True):
        st.rerun()

    # Load analytics data
    try:
        res_analytics = requests.get(f"{BACKEND_URL}/analytics")
        if res_analytics.status_code == 200:
            analytics = res_analytics.json()
            
            # Render KPI Metric Cards
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{analytics['total_questions']}</div>
                    <div class='metric-lbl'>Total Queries</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi_col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{analytics['success_rate']}%</div>
                    <div class='metric-lbl'>Success Rate</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi_col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{analytics['avg_response_time_ms']} ms</div>
                    <div class='metric-lbl'>Avg Latency</div>
                </div>
                """, unsafe_allow_html=True)
                
            with kpi_col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{analytics['unanswered_questions_count']}</div>
                    <div class='metric-lbl'>Unanswered Queries</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Charts Section
            st.markdown("---")
            chart_col1, chart_col2 = st.columns([1.5, 1])
            
            # Prepare Daily Usage Data
            daily_data = analytics.get("daily_usage", [])
            df_daily = pd.DataFrame(daily_data)
            
            with chart_col1:
                st.markdown("#### Query Volume Trend")
                if not df_daily.empty:
                    # Daily Volume Bar Chart
                    fig_volume = px.bar(
                        df_daily,
                        x="date",
                        y="count",
                        labels={"date": "Date", "count": "Number of Queries"},
                        color_discrete_sequence=["#FF9900"],
                        template="plotly_dark"
                    )
                    fig_volume.update_layout(
                        margin=dict(l=20, r=20, t=10, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)
                else:
                    st.info("No daily usage data available. Run queries to populate charts.")
            
            with chart_col2:
                st.markdown("#### Retrieval Success Distribution")
                success_count = analytics['total_questions'] - analytics['unanswered_questions_count']
                fail_count = analytics['unanswered_questions_count']
                
                if analytics['total_questions'] > 0:
                    # Donut Chart for Success Rate
                    labels = ['Answered (In Context)', 'Unanswered (Out of Context)']
                    values = [success_count, fail_count]
                    
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.5,
                        marker_colors=['#10b981', '#ef4444'],
                        textinfo='percent+value'
                    )])
                    fig_donut.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("No query logs available.")
            
            # Latency Trend & Top Queries
            st.markdown("---")
            bottom_col1, bottom_col2 = st.columns([1, 1])
            
            with bottom_col1:
                st.markdown("#### Top 10 Frequently Asked Queries")
                top_queries = analytics.get("top_queries", [])
                if top_queries:
                    df_top = pd.DataFrame(top_queries)
                    df_top.columns = ["Question", "Frequency", "Avg Latency (ms)"]
                    st.dataframe(df_top, use_container_width=True, hide_index=True)
                else:
                    st.info("No query logs available.")
                    
            with bottom_col2:
                st.markdown("#### Retrieval Quality Statistics")
                avg_chunks = analytics.get("retrieved_chunks_stats", {}).get("avg_chunks_retrieved", 0.0)
                st.markdown(f"**Average Chunks Retrieved per Query:** `{avg_chunks}` chunks")
                st.markdown("""
                - **Top-K Retrieval Setting:** Top 5 chunks are retrieved for each search query.
                - **Similarity Threshold Safety:** If the cosine similarity score of the top chunk is below 55%, the pipeline blocks LLM invocation to prevent hallucinations.
                - **Contextual Chunk Injection:** Chunks are enriched with section titles and page locations to provide optimal legal grounding for the generation phase.
                """)
                
        else:
            st.error("Failed to load analytics data from API server.")
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.info("Please make sure the backend server is running and database queries are populated.")
