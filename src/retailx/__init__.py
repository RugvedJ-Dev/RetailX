import os
import json
import asyncio
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# If the MCP server is in a separate project directory,
# configure MCP_PROJECT_DIR in your .env file.
MCP_PROJECT_DIR = Path(
    os.getenv("MCP_PROJECT_DIR", str(PROJECT_ROOT))
).resolve()

MCP_SERVER_PATH = (
    MCP_PROJECT_DIR
    / "src"
    / "temp_major_mcp"
    / "__init__.py"
)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b",
)

CURRENCY_SYMBOL = os.getenv(
    "CURRENCY_SYMBOL",
    "₹",
)


# ==================================================
# STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetric"] {
            background: #f7f9fc;
            border: 1px solid #e5eaf1;
            padding: 18px;
            border-radius: 12px;
        }

        .main-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .sub-title {
            color: #718096;
            margin-top: 5px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# MCP CONNECTION
# ==================================================

@st.cache_resource
def load_mcp_tools():

    if not MCP_SERVER_PATH.exists():

        raise FileNotFoundError(
            f"MCP server not found: {MCP_SERVER_PATH}"
        )

    client = MultiServerMCPClient(
        {
            "sales": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(MCP_PROJECT_DIR),
                    "fastmcp",
                    "run",
                    "src/temp_major_mcp/__init__.py",
                ],
            }
        }
    )

    tools = asyncio.run(client.get_tools())

    if not tools:
        raise RuntimeError("No MCP tools were discovered.")

    return client, tools


# ==================================================
# GROQ LLM
# ==================================================

@st.cache_resource
def load_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from your .env file."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0.2,
        api_key=api_key,
    )


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def format_currency(value):

    return f"{CURRENCY_SYMBOL}{float(value):,.2f}"


def normalize_tool_result(result):

    if isinstance(result, dict):
        return result

    if isinstance(result, str):

        try:
            return json.loads(result)

        except json.JSONDecodeError:
            return {"result": result}

    if isinstance(result, list):

        text_parts = []

        for item in result:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

            elif hasattr(item, "text"):
                text_parts.append(item.text)

        if text_parts:

            combined = "\n".join(text_parts)

            try:
                return json.loads(combined)

            except json.JSONDecodeError:
                return {"result": combined}

    return {"result": str(result)}


def get_message_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))

        return "\n".join(parts)

    return str(content) if content else ""


def trend_to_dataframe(trend, start=None, end=None):
    """Turn a get_sales_trend result into a daily dataframe
    indexed by date (zero-filled for days without sales)."""

    records = trend.get("trend", [])

    if not records:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    start = start or trend.get("start_date") or df["date"].min()
    end = end or trend.get("end_date") or df["date"].max()

    all_dates = pd.date_range(start=start, end=end, freq="D")

    return (
        df.set_index("date")
        .reindex(all_dates, fill_value=0)
        .rename_axis("date")
    )


def render_trend_chart(trend, start=None, end=None, height=320):
    """Render a revenue line chart with st.line_chart."""

    df = trend_to_dataframe(trend, start, end)

    if df is None:
        st.info("No sales data available for this period.")
        return

    st.caption(
        f"Daily revenue: {trend.get('start_date', start)} "
        f"to {trend.get('end_date', end)}"
    )

    st.line_chart(
        df[["revenue"]],
        height=height,
        y_label=f"Revenue ({CURRENCY_SYMBOL})",
    )


def render_sales_chart(data, height=320):
    """Show filtered sales (e.g. 'sales <= 70') as a line chart
    of each sale's amount, plus the matching records table."""

    records = data.get("sales", [])

    if not records:
        st.info("No sales matched your request.")
        return

    df = pd.DataFrame(records)

    # Oldest first so the line reads left to right in time.
    df = df.sort_values(["sale_date", "id"]).reset_index(drop=True)

    st.caption(f"{len(df)} matching sale(s)")

    if len(df) >= 2:

        chart_df = df.copy()

        chart_df["sale"] = (
            chart_df["sale_date"].astype(str)
            + " #"
            + chart_df["id"].astype(str)
        )

        st.line_chart(
            chart_df.set_index("sale")[["total_amount"]],
            height=height,
            y_label=f"Sale amount ({CURRENCY_SYMBOL})",
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# CHAT INITIALIZATION
# ==================================================

def initialize_chat():

    if "messages" not in st.session_state:

        st.session_state.messages = [

            SystemMessage(
                content=f"""
You are an intelligent AI sales management assistant.

Today's date is {date.today().isoformat()}.

Your responsibilities:

1. Record sales when the user asks you to add or record a sale.
2. Retrieve sales records when requested.
3. Analyze revenue, transactions and products.
4. Answer questions using actual database records.
5. Never invent sales figures or transaction records.
6. Use MCP tools whenever a question requires sales data.
7. For a new sale, collect product, quantity and unit price
   if any of these are missing.
8. If the user does not provide a sale date, use today's date.
9. Clearly mention the date range when discussing revenue.
10. Keep responses professional and easy to understand.
11. When the user asks for a chart, graph, trend or any
    visualization of sales/revenue over time, call the
    get_sales_trend tool for the requested date range. The app
    automatically draws a line chart from the result, so do not
    draw ASCII charts or tables of every day. Just add a short
    summary (total, highest day, notable pattern) after the chart.
12. For requests filtered by amount, such as "sales of 70 rupees
    or less" or "sales above 500", call get_sales with min_amount
    and/or max_amount (these filter the total amount of a sale).
    "Less than or equal to 70" means max_amount=70; "at least 70"
    means min_amount=70. Only use min_unit_price/max_unit_price
    when the user clearly says "per unit" or "unit price". The app
    automatically shows a line chart and table of the matching
    sales, so just summarize them briefly (count, total, range).
    The amount filters can be combined with dates and products,
    and get_sales_trend and sales_summary also accept
    min_amount and max_amount.

Available MCP tools:

- add_sale
- get_sales
- sales_summary
- get_sales_trend
- top_products

Currency: {CURRENCY_SYMBOL}

Never claim that a sale was saved unless the add_sale tool
successfully confirms it.
"""
            )

        ]


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

def display_chat_history():

    for message in st.session_state.messages:

        if isinstance(message, HumanMessage):

            with st.chat_message("user"):
                st.markdown(message.content)

        elif isinstance(message, AIMessage):

            text = get_message_text(message.content)

            if text:

                with st.chat_message("assistant"):
                    st.markdown(text)

        elif (
            isinstance(message, ToolMessage)
            and message.name == "get_sales"
        ):

            try:
                data = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue

            if isinstance(data, dict) and "sales" in data:

                with st.chat_message("assistant"):
                    render_sales_chart(data)

        elif (
            isinstance(message, ToolMessage)
            and message.name == "get_sales_trend"
        ):

            # Re-draw the chart from the stored tool result.

            try:
                data = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue

            if isinstance(data, dict) and "trend" in data:

                with st.chat_message("assistant"):
                    render_trend_chart(data)


# ==================================================
# ASYNC MCP TOOL EXECUTION
# ==================================================

async def execute_mcp_tool(tool, arguments):

    result = await tool.ainvoke(arguments)

    return normalize_tool_result(result)


# ==================================================
# AI AGENT EXECUTION
# ==================================================

def run_agent(user_input, tools, llm):

    messages = st.session_state.messages

    messages.append(
        HumanMessage(content=user_input)
    )

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    model_with_tools = llm.bind_tools(tools)

    with st.chat_message("assistant"):

        # Allow multiple tool-call cycles.

        for step in range(6):

            streamed_messages = []

            def token_stream():

                full_response = None

                for chunk in model_with_tools.stream(messages):

                    if full_response is None:
                        full_response = chunk

                    else:
                        full_response = full_response + chunk

                    content = chunk.content

                    if isinstance(content, str) and content:
                        yield content

                streamed_messages.append(full_response)

            with st.spinner("Thinking..."):

                st.write_stream(token_stream())

            if not streamed_messages or streamed_messages[0] is None:

                st.error(
                    "The model did not return a response."
                )

                return

            response = streamed_messages[0]

            # The streamed response is stored in history as-is.
            # Do NOT append a second AIMessage afterwards,
            # or the reply will show up twice on the next rerun.

            messages.append(response)

            # No tool calls means the assistant has finished.

            if not response.tool_calls:
                break

            # Execute every requested MCP tool.

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call["args"]

                tool_call_id = tool_call["id"]

                result = None

                with st.status(
                    f"Executing: {tool_name}",
                    expanded=True,
                ) as status:

                    st.write(
                        "Connecting to the MCP server..."
                    )

                    st.write("Tool arguments:")

                    st.json(tool_args)

                    try:

                        selected_tool = tool_map.get(
                            tool_name
                        )

                        if selected_tool is None:

                            raise ValueError(
                                f"Unknown MCP tool: {tool_name}"
                            )

                        # MCP tools are asynchronous.
                        # Never use selected_tool.invoke() here.

                        result = asyncio.run(
                            execute_mcp_tool(
                                selected_tool,
                                tool_args,
                            )
                        )

                        st.write(
                            "Tool executed successfully."
                        )

                        st.json(result)

                        status.update(
                            label=f"{tool_name} completed",
                            state="complete",
                            expanded=False,
                        )

                        tool_content = json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        )

                    except Exception as error:

                        status.update(
                            label=f"{tool_name} failed",
                            state="error",
                            expanded=True,
                        )

                        tool_content = json.dumps(
                            {
                                "success": False,
                                "error": str(error),
                            },
                            ensure_ascii=False,
                        )

                # Draw the chart right in the chat when the
                # trend tool returned data.

                if (
                    tool_name == "get_sales_trend"
                    and isinstance(result, dict)
                    and "trend" in result
                ):
                    render_trend_chart(result)

                elif (
                    tool_name == "get_sales"
                    and isinstance(result, dict)
                    and "sales" in result
                ):
                    render_sales_chart(result)

                # Preserve the corresponding tool-call ID.
                # The name lets display_chat_history re-draw charts.

                messages.append(
                    ToolMessage(
                        content=tool_content,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        else:

            warning_text = (
                "I could not complete the request "
                "within the allowed tool-call steps."
            )

            st.warning(warning_text)

            messages.append(
                AIMessage(content=warning_text)
            )


# ==================================================
# DASHBOARD DATA RETRIEVAL
# ==================================================

async def load_dashboard_data(
    tools,
    start_date,
    end_date,
):

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    async def execute_tool(name, arguments):

        tool = tool_map.get(name)

        if tool is None:

            raise ValueError(
                f"MCP tool '{name}' was not found."
            )

        result = await tool.ainvoke(arguments)

        return normalize_tool_result(result)

    start = start_date.isoformat()
    end = end_date.isoformat()

    # Execute all four tools concurrently.

    summary, trend, sales, products = await asyncio.gather(

        execute_tool(
            "sales_summary",
            {
                "start_date": start,
                "end_date": end,
            },
        ),

        execute_tool(
            "get_sales_trend",
            {
                "start_date": start,
                "end_date": end,
            },
        ),

        execute_tool(
            "get_sales",
            {
                "start_date": start,
                "end_date": end,
                "limit": 5000,
            },
        ),

        execute_tool(
            "top_products",
            {
                "start_date": start,
                "end_date": end,
                "limit": 5,
            },
        ),

    )

    return summary, trend, sales, products


# ==================================================
# LOAD BACKEND
# ==================================================

try:

    _, mcp_tools = load_mcp_tools()

    llm = load_llm()

except Exception as error:

    st.error("Backend connection failed.")

    st.exception(error)

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("Sales Intelligence")

    st.caption(
        "AI-powered sales management"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "AI Assistant",
        ],
    )

    st.divider()

    st.caption("Database")

    st.success("SQLite configured")

    st.caption("MCP Server")

    st.success("Local STDIO")

    st.caption(f"Model: {MODEL_NAME}")

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.pop(
            "messages",
            None,
        )

        st.rerun()


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Sales Dashboard</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-title">'
        'Monitor your business performance and sales activity.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    today = date.today()

    default_start = today.replace(day=1)

    col1, col2, col3 = st.columns(
        [1, 1, 2]
    )

    with col1:

        start_date = st.date_input(
            "Start date",
            value=default_start,
        )

    with col2:

        end_date = st.date_input(
            "End date",
            value=today,
        )

    with col3:

        st.write("")
        st.write("")

        st.button(
            "Refresh dashboard",
            use_container_width=True,
        )

    if start_date > end_date:

        st.error(
            "Start date cannot be after end date."
        )

        st.stop()

    # --------------------------------------------------
    # FETCH DASHBOARD DATA
    # --------------------------------------------------

    try:

        with st.spinner(
            "Fetching sales data..."
        ):

            summary, trend, sales, products = asyncio.run(
                load_dashboard_data(
                    mcp_tools,
                    start_date,
                    end_date,
                )
            )

        st.divider()

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Total Revenue",
            format_currency(
                summary.get("total_revenue", 0)
            ),
        )

        metric2.metric(
            "Transactions",
            summary.get("total_transactions", 0),
        )

        metric3.metric(
            "Units Sold",
            summary.get("units_sold", 0),
        )

        metric4.metric(
            "Average Transaction",
            format_currency(
                summary.get("average_transaction", 0)
            ),
        )

        st.divider()

        # --------------------------------------------------
        # REVENUE TREND (st.line_chart)
        # --------------------------------------------------

        st.subheader("Revenue Trend")

        trend_records = trend.get(
            "trend",
            [],
        )

        if trend_records:

            trend_df = pd.DataFrame(
                trend_records
            )

            trend_df["date"] = pd.to_datetime(
                trend_df["date"]
            )

            # Include dates with zero sales.

            all_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq="D",
            )

            trend_df = (
                trend_df
                .set_index("date")
                .reindex(
                    all_dates,
                    fill_value=0,
                )
                .rename_axis("date")
            )

            # st.line_chart uses the index (date) as the x-axis.
            st.line_chart(
                trend_df[["revenue"]],
                height=380,
                y_label=f"Revenue ({CURRENCY_SYMBOL})",
            )

        else:

            st.info(
                "No sales data available for this period."
            )

        st.divider()

        # --------------------------------------------------
        # TOP PRODUCTS
        # --------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.subheader("Top Products")

            product_records = products.get(
                "top_products",
                [],
            )

            if product_records:

                product_df = pd.DataFrame(
                    product_records
                )

                st.dataframe(
                    product_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No product data available."
                )

        # --------------------------------------------------
        # SALES OVERVIEW
        # --------------------------------------------------

        with right:

            st.subheader("Sales Overview")

            st.write(
                f"**Period:** {start_date} to {end_date}"
            )

            st.write(
                f"**Revenue:** "
                f"{format_currency(summary.get('total_revenue', 0))}"
            )

            st.write(
                f"**Units sold:** "
                f"{summary.get('units_sold', 0)}"
            )

            st.write(
                f"**Transactions:** "
                f"{summary.get('total_transactions', 0)}"
            )

        st.divider()

        # --------------------------------------------------
        # SALES RECORDS
        # --------------------------------------------------

        st.subheader("Sales Records")

        sales_records = sales.get(
            "sales",
            [],
        )

        if sales_records:

            sales_df = pd.DataFrame(
                sales_records
            )

            st.dataframe(
                sales_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No sales records found."
            )

    except Exception as error:

        st.error(
            "Could not load dashboard."
        )

        st.exception(error)


# ==================================================
# AI ASSISTANT
# ==================================================

elif page == "AI Assistant":

    st.markdown(
        '<p class="main-title">AI Sales Assistant</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-title">'
        'Ask questions, record sales and analyze your business.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    initialize_chat()

    display_chat_history()

    user_input = st.chat_input(
        "Ask about your sales or record a new sale..."
    )

    if user_input:

        with st.chat_message("user"):

            st.markdown(user_input)

        try:

            run_agent(
                user_input,
                mcp_tools,
                llm,
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )

            st.exception(error)
