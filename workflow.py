from langgraph.graph import StateGraph, END
from agent_state import AgentState
from agent_nodes import (
    get_current_user,
    check_relevance,
    convert_nl_to_sql,
    execute_sql,
    generate_human_readable_answer,
    regenerate_query,
    generate_funny_response,
    end_max_iterations
)

# ---------------------------------------------------
# Create and configure the LangGraph workflow
# ---------------------------------------------------
def create_workflow():
    
    # Define router functions
    def relevance_router(state: AgentState):
        if state["relevance"].lower() == "relevant":
            return "convert_to_sql"
        else:
            return "generate_funny_response"

    def check_attempts_router(state: AgentState):
        if state["attempts"] < 3:
            return "convert_to_sql"
        else:
            return "end_max_iterations"

    def execute_sql_router(state: AgentState):
        if not state.get("sql_error", False):
            return "generate_human_readable_answer"
        else:
            return "regenerate_query"

    # Create workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("get_current_user", get_current_user)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("convert_to_sql", convert_nl_to_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
    workflow.add_node("regenerate_query", regenerate_query)
    workflow.add_node("generate_funny_response", generate_funny_response)
    workflow.add_node("end_max_iterations", end_max_iterations)

    # Define workflow edges
    workflow.add_edge("get_current_user", "check_relevance")

    workflow.add_conditional_edges(
        "check_relevance",
        relevance_router,
        {
            "convert_to_sql": "convert_to_sql",
            "generate_funny_response": "generate_funny_response",
        },
    )

    workflow.add_edge("convert_to_sql", "execute_sql")

    workflow.add_conditional_edges(
        "execute_sql",
        execute_sql_router,
        {
            "generate_human_readable_answer": "generate_human_readable_answer",
            "regenerate_query": "regenerate_query",
        },
    )

    workflow.add_conditional_edges(
        "regenerate_query",
        check_attempts_router,
        {
            "convert_to_sql": "convert_to_sql",
            "max_iterations": "end_max_iterations",
        },
    )

    # End nodes
    workflow.add_edge("generate_human_readable_answer", END)
    workflow.add_edge("generate_funny_response", END)
    workflow.add_edge("end_max_iterations", END)

    # Set entry point
    workflow.set_entry_point("get_current_user")
    
    return workflow.compile()

# Create the application
app = create_workflow()