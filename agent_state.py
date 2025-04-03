from typing_extensions import TypedDict
from pydantic import BaseModel, Field

#---------------------------------------------------
# Define the AgentState type
#---------------------------------------------------
class AgentState(TypedDict):
    """State object passed between nodes in the workflow"""
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    current_user: str
    attempts: int
    relevance: str
    sql_error: bool

#---------------------------------------------------
# Output models for structured LLM responses
#---------------------------------------------------
class GetCurrentUser(BaseModel):
    current_user: str = Field(
        description="The name of the current user based on the provided user ID."
    )

#---------------------------------------------------
# Output models for structured LLM responses
#---------------------------------------------------
class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'."
    )

#---------------------------------------------------
# Output models for structured LLM responses
#---------------------------------------------------
class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The SQL query corresponding to the user's natural language question."
    )

#---------------------------------------------------
# Output models for structured LLM responses
#---------------------------------------------------
class RewrittenQuestion(BaseModel):
    question: str = Field(
        description="The rewritten question."
    )