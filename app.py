import streamlit as st
import os 
from foot_app.backend import DB_PATH, SYSTEM_PROMPT, VISION_TOOL_PROMPT, build_afcon_db
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
vision_supported = model_choice == "Gemini 3 Flash"
if "vision_enabled" not in st.session_state:
    st.session_state.vision_enabled = vision_supported

vision_enabled = st.sidebar.checkbox(
    "Enable vision analysis",
    key="vision_enabled",
    help="Uses the selected model if it supports vision, or a separate Gemini key if not.",
)

vision_model_choice = None
vision_api_key = ""
if vision_enabled and not vision_supported:
    st.sidebar.subheader("Vision Model")
    vision_model_choice = st.sidebar.selectbox(
        "Vision model",
        ["Gemini 3 Flash"],
        help="Used only for image analysis; the main agent stays on the selected model.",
    )
    vision_api_key = st.sidebar.text_input(
        "Vision API key",
        type="password",
        help="Google AI Studio key for the vision model.",
    )
elif vision_supported:
    st.sidebar.caption("Vision analysis will use the selected Gemini model.")
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
        os.environ["GOOGLE_API_KEY"] = api_key
        planner = executor = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
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

    vision_model = None
    if vision_enabled:
        if vision_supported:
            vision_model = executor
        else:
            if not vision_api_key.strip():
                st.error("Please enter the Vision API key in the sidebar.")
                st.stop()
            os.environ["GOOGLE_API_KEY"] = vision_api_key
            vision_model = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                temperature=temperature,
                max_retries=6,
            )

    vision_tool_available = bool(vision_model)

    planner_tools = "table_columns, sql_query, sql_to_df, python_viz"
    vision_plan_rule = ""
    if vision_tool_available:
        planner_tools += ", vision_analyze_plot"
        vision_plan_rule = "- If a plot image needs describing, plan to call vision_analyze_plot after python_viz using the saved plot path.\n"

    plan_prompt = (
        f"""You are the PLANNER for a football analytics agent working on AFCON 2023 StatsBomb Open Data in SQLite.
        Your job: output a short, executable plan the EXECUTOR will follow using tools:
        {planner_tools}.

        CRITICAL RULES:
        - Do NOT call tools. Do NOT write final SQL results. Only plan steps.
        - Prefer the fewest tool calls possible.
        - If the question is match-specific, plan to resolve match_id via the games table FIRST.
        - Only plan schema checks (table_columns or column_map queries) if a required column is uncertain.
        - Avoid selecting raw_json unless explicitly required.
        - For event sequences: include ORDER BY event_index ASC.
        - For plots: fetch ONLY needed columns; plan 1 sql_to_df + 1 python_viz whenever possible.
        {vision_plan_rule}
        OUTPUT FORMAT (STRICT): Return JSON only, no extra text.
        Schema:
        {{
        "task": "one sentence",
        "needs_plot": true/false,
        "match_resolution": {{
            "needed": true/false,
            "strategy": "how to find match_id(s) from games",
            "disambiguation": "what to show if multiple matches"
        }},
        "schema_checks": [
            {{"when": "condition", "action": "table_columns('events') OR query column_map", "target": "which fields"}}
        ],
        "sql_steps": [
            {{
            "tool": "sql_query | sql_to_df",
            "purpose": "what this query returns",
            "sql_outline": "SQL skeleton with key filters/joins (no SELECT *)",
            "expected_columns": ["col1","col2"],
            "df_name": "only if sql_to_df"
            }}
        ],
        "viz_plan": {{
            "plot_type": "none | shot_map | pass_map | heatmap | network | sequence_map | bar | line",
            "pitch": true/false,
            "required_fields": ["x","y","end_x","end_y","label","color_by"],
            "grouping": "if any",
            "annotations": "if any"
        }},
        "validation": [
            "checks on row_count, distinct match_id, entity matches, event types, missing locations"
        ],
        "final_response": "how to present the answer (table/bullets + key counts + match context)"
        }}
        """
        f"USER QUESTION:\n{prompt.strip()}"
    )
    plan_text = str(planner.invoke(plan_prompt).content).strip()[:2000]
    st.session_state.last_plan = plan_text

    tools = get_tools(vision_model=vision_model, enable_vision_tool=vision_tool_available)

    system_prompt = SYSTEM_PROMPT
    if vision_tool_available:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{VISION_TOOL_PROMPT}"

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
            system_prompt=system_prompt,
            user_prompt=injected_user_prompt,
            max_steps=50,
            on_event=on_event,
        )

    status_box.empty()

    # Ensure state matches final outputs
    st.session_state.history = outputs
    st.session_state.last_trace = trace

    st.rerun()
