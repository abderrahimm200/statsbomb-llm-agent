import streamlit as st

from foot_app.backend import DB_PATH, SYSTEM_PROMPT, build_afcon_db
from foot_app.tools import get_tools
from foot_app.agent_loop import run_tool_calling_loop

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


st.set_page_config(page_title="AFCON StatsBomb Agent", layout="wide")
st.title("AFCON 2023 StatsBomb SQL + Viz Agent")

# ----------------------------
# Session state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None

# ----------------------------
# Sidebar: model + key
# ----------------------------
st.sidebar.header("Model")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Gemini 3 Flash", "DeepSeek V3.2"],
)

api_key = st.sidebar.text_input(
    "API key",
    type="password",
    help="Use a Google AI Studio key for Gemini, or a DeepSeek API key for DeepSeek.",
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
st.sidebar.divider()

# ----------------------------
# DB status + build button
# ----------------------------
db_ready = DB_PATH.exists()
if db_ready:
    st.sidebar.success(f"DB ready: {DB_PATH}")
else:
    st.sidebar.warning("DB not found. Build it once (cached on disk).")

if st.sidebar.button("Build / Rebuild database", disabled=False):
    prog = st.sidebar.progress(0, text="Starting...")

    def on_prog(p):
        frac = p.i / max(1, p.total)
        prog.progress(frac, text=f"{p.stage} ({p.i}/{p.total})")

    with st.spinner("Building SQLite database from StatsBomb Open Data..."):
        build_afcon_db(force=True, progress_cb=on_prog)

    st.sidebar.success("Database built.")
    st.rerun()

st.sidebar.divider()

# ----------------------------
# Main input
# ----------------------------
prompt = st.text_area(
    "Your question",
    height=140,
    placeholder="Example: For Morocco vs South Africa, where did Morocco lose possession most often in midfield? Visualize it.",
)

run_clicked = st.button("Generate", type="primary", disabled=(not prompt.strip()))

# ----------------------------
# Render existing history (completed)
# ----------------------------
def render_item(item):
    if item["type"] == "assistant_text":
        with st.chat_message("assistant"):
            st.write(item["text"])
    elif item["type"] == "plot":
        st.image(item["image_bytes"])

for item in st.session_state.history:
    render_item(item)

# Debug (hidden by default)
with st.expander("Debug trace (tool calls, args, outputs)", expanded=False):
    if st.session_state.last_plan:
        st.markdown("**Planner output (hidden from main UI):**")
        st.code(st.session_state.last_plan)

    if st.session_state.last_trace is None:
        st.write("No trace yet.")
    else:
        st.json(st.session_state.last_trace)

# ----------------------------
# Run agent (real-time)
# ----------------------------
if run_clicked:
    # Clear any previous displayed messages/plots immediately
    st.session_state.history = []
    st.session_state.last_trace = None
    st.session_state.last_plan = None

    if not DB_PATH.exists():
        st.error("Database not built yet. Click 'Build / Rebuild database' in the sidebar.")
        st.stop()

    if not api_key.strip():
        st.error("Please enter the API key in the sidebar.")
        st.stop()

    plan_text = None

    # Create executor (and planner for DeepSeek)
    if model_choice == "Gemini 3 Flash":
        executor = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            api_key=api_key,
            temperature=temperature,
            max_retries=6,
        )
    else:
        planner = ChatOpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            model="deepseek-reasoner",
            temperature=0.2,
            extra_body={"thinking": {"type": "enabled"}},
        )
        executor = ChatOpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            model="deepseek-chat",
            temperature=temperature,
        )

        plan_prompt = (
            "Create a concise numbered plan to answer the user's question using the available tools "
            "(table_columns, sql_query/sql_to_df, python_viz). "
            "DO NOT call tools. "
            "Include: which table(s) to inspect, the SQL strategy, the dataframe columns needed, "
            "and what plot to generate. Keep it short.\n\n"
            f"USER QUESTION:\n{prompt.strip()}"
        )
        plan_text = str(planner.invoke(plan_prompt).content).strip()[:2000]
        st.session_state.last_plan = plan_text

    tools = get_tools()

    injected_user_prompt = prompt.strip()
    if plan_text:
        injected_user_prompt = (
            "You must follow the plan below. Do not mention the plan unless asked.\n\n"
            "PLAN:\n"
            f"{plan_text}\n\n"
            "USER QUESTION:\n"
            f"{prompt.strip()}"
        )

    # Real-time UI placeholders
    status_box = st.empty()
    live_container = st.container()

    def on_event(event: dict):
        # Status updates (shown live, not stored in history)
        if event.get("type") == "status":
            status_box.info(event.get("text", "…"))
            return

        # Visible outputs: store + render immediately
        st.session_state.history.append(event)
        with live_container:
            render_item(event)

    with st.spinner("Running agent..."):
        outputs, trace = run_tool_calling_loop(
            model=executor,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=injected_user_prompt,
            max_steps=50,
            on_event=on_event,
        )

    status_box.empty()

    # Ensure state matches final outputs
    st.session_state.history = outputs
    st.session_state.last_trace = trace

    st.rerun()
