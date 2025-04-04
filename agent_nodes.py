import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text
from langchain_core.runnables import RunnableConfig
from model_factory import ModelFactory
from db_utils import SessionLocal, get_database_schema
from agent_state import AgentState, GetCurrentUser, CheckRelevance, ConvertToSQL, RewrittenQuestion
from model.user import User
from util.logger import log 

#------------------------------------------------------------------------
# Create a chat prompt template for checking relevance
#------------------------------------------------------------------------
def create_chat_prompt(system_prompt, human_prompt):
    check_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])    
    return check_prompt

#-------------------------------------------------------------------------
# Retrieve current user information based on user ID in config
#-------------------------------------------------------------------------
def get_current_user(state: AgentState, config: RunnableConfig):
    log.info("Retrieving the current user based on user ID.")
    user_id = config["configurable"].get("current_user_id", None)
    
    if not user_id:
        state["current_user"] = "User not found"
        log.info("No user ID provided in the configuration.")
        return state

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == int(user_id)).first()
        if user:
            state["current_user"] = user.name
            log.info(f"Current user set to: {state['current_user']}")
        else:
            state["current_user"] = "User not found"
            log.info("User not found in the database.")
    except Exception as e:
        state["current_user"] = "Error retrieving user"
        log.info(f"Error retrieving user: {str(e)}")
    finally:
        session.close()
    
    return state


#-------------------------------------------------------------------------
# Determine if user question is relevant to the database schema
#-------------------------------------------------------------------------
def check_relevance(state: AgentState, config: RunnableConfig):
    question = state["question"]
    schema = get_database_schema()
    log.info(f"Checking relevance of the question: {question}")
    
    # Define the system prompt with proper formatting
    system_prompt = """
    You are an assistant that determines whether a given question is related to the following database schema.    
    Schema:
    {schema}    
    Respond with only "relevant" or "not_relevant".
    """.format(schema=schema).strip()
    
    # Create the human message
    human_prompt = f"Question: {question}"
    
    # Build the prompt template
    chat_prompt = create_chat_prompt(system_prompt, human_prompt)
    
    # Configure the LLM with structured output
    llm = ModelFactory.get_model(provider="openai",model_name="gpt-4o")
    structured_llm = llm.with_structured_output(CheckRelevance)
    
    # Create the chain and execute
    relevance_checker = chat_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    
    # Update state with result
    state["relevance"] = relevance.relevance
    log.info(f"Relevance determined: {state['relevance']}")
    
    return state


#-------------------------------------------------------------------------
# Convert natural language question to SQL query
#-------------------------------------------------------------------------
def convert_nl_to_sql(state: AgentState, config: RunnableConfig):
    question = state["question"]
    current_user = state["current_user"]
    schema = get_database_schema()
    log.info(f"Converting question to SQL for user '{current_user}': {question}")
    
    # Define the system prompt with proper formatting
    system_prompt = """
    You are an assistant that converts natural language questions into SQL queries based on the following schema:    
    {schema}    
    The current user is '{current_user}'. Ensure that all query-related data is scoped to this user.    
    Provide only the SQL query without any explanations. Alias columns appropriately to match the expected keys in the result.
    For example, alias 'food.name' as 'food_name' and 'food.price' as 'price'.
    """.format(schema=schema, current_user=current_user).strip()
    
    human_prompt = f"Question: {question}"
     
    # Build the prompt template
    chat_prompt = create_chat_prompt(system_prompt, human_prompt)
    
    # Configure the LLM with structured output
    llm = ModelFactory.get_model(provider="openai",model_name="gpt-4o")
    structured_llm = llm.with_structured_output(ConvertToSQL)
    
    # Create the chain and execute
    sql_generator = chat_prompt | structured_llm
    result = sql_generator.invoke({"question": question})
    
    # Update state with result
    state["sql_query"] = result.sql_query
    log.info(f"Generated SQL query: {state['sql_query']}")
    
    return state


#-------------------------------------------------------------------------
# Execute SQL query against the database
#-------------------------------------------------------------------------
def execute_sql(state: AgentState):
    sql_query = state["sql_query"].strip()
    session = SessionLocal()
    log.info(f"Executing SQL query: {sql_query}")
    
    try:
        result = session.execute(text(sql_query))
        
        if sql_query.lower().startswith("select"):
            rows = result.fetchall()
            columns = result.keys()
            
            if rows:
                header = ", ".join(columns)
                state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                log.info(f"Raw SQL Query Result: {state['query_rows']}")
                
                # Format the result for readability
                data = "; ".join([
                    f"{row.get('food_name', row.get('name'))} for ${row.get('price', row.get('food_price'))}" 
                    for row in state["query_rows"]
                ])
                formatted_result = f"{header}\n{data}"
            else:
                state["query_rows"] = []
                formatted_result = "No results found."
                
            state["query_result"] = formatted_result
            state["sql_error"] = False
            log.info("SQL SELECT query executed successfully.")
        else:
            session.commit()
            state["query_result"] = "The action has been successfully completed."
            state["sql_error"] = False
            log.info("SQL command executed successfully.")
            
            
    except Exception as e:
        state["query_result"] = f"Error executing SQL query: {str(e)}"
        state["sql_error"] = True
        log.error(f"Error executing SQL query: {str(e)}")
    finally:
        session.close()
        
    return state


#-------------------------------------------------------------------------
# Generate natural language response from SQL query results
#-------------------------------------------------------------------------
def generate_human_readable_answer(state: AgentState):
    sql = state["sql_query"]
    result = state["query_result"]
    current_user = state["current_user"]
    query_rows = state.get("query_rows", [])
    sql_error = state.get("sql_error", False)
    log.info("Generating a human-readable answer.")
    
    # Define the base system prompt
    system_prompt = """
    You are an assistant that converts SQL query results into clear, natural language responses 
    without including any identifiers like order IDs. Start the response with a friendly greeting 
    that includes the user's name.
    """.strip()
    
    # Handle different cases with appropriate prompts
    if sql_error:
        # Error case
        human_prompt = f"""
        SQL Query:
        {sql}
        
        Result:
        {result}
        
        Formulate a clear and understandable error message in a single sentence, 
        starting with 'Hello {current_user},' informing them about the issue.
        """.strip()
        
    elif sql.lower().startswith("select"):
        if not query_rows:
            # No results case
            human_prompt = f"""
            SQL Query:
            {sql}
            
            Result:
            {result}
            
            Formulate a clear and understandable answer to the original question in a single sentence, 
            starting with 'Hello {current_user},' and mention that there are no orders found.
            """.strip()
            
        else:
            # Displaying orders case
            human_prompt = f"""
            SQL Query:
            {sql}
            
            Result:
            {result}
            
            Formulate a clear and understandable answer to the original question in a single sentence, 
            starting with 'Hello {current_user},' and list each item ordered along with its price.             
            """.strip()
            
    else:
        # Non-select query confirmation
        human_prompt = f"""
        SQL Query:
        {sql}
        
        Result:
        {result}
        
        Formulate a clear and understandable confirmation message in a single sentence, 
        starting with 'Hello {current_user},' confirming that your request has been successfully processed.
        """.strip()

    # Create the prompt template
    generate_prompt = create_chat_prompt(system_prompt, human_prompt)
    
    # Configure and execute the LLM
    llm = ModelFactory.get_model(provider="openai",model_name="gpt-4o")
    human_response = generate_prompt | llm | StrOutputParser()
    answer = human_response.invoke({})
    
    # Update state with result
    state["query_result"] = answer
    log.info("Generated human-readable answer.")
    
    return state


#-------------------------------------------------------------------------
# Rewrite the question to improve SQL generation on failure
#-------------------------------------------------------------------------
def regenerate_query(state: AgentState):
    question = state["question"]
    log.info("Regenerating the SQL query by rewriting the question.")
    
    # Define the system prompt
    system_prompt = """
    You are an assistant that reformulates an original question to enable more precise SQL queries. 
    Ensure that all necessary details, such as table joins, are preserved to retrieve complete and accurate data.
    """.strip()
    
    # Define the human prompt
    human_prompt = f"""
    Original Question: {question}
    Reformulate the question to enable more precise SQL queries, ensuring all necessary details are preserved.
    """.strip()
    
    # Create the prompt template
    rewrite_prompt = create_chat_prompt(system_prompt, human_prompt)
    
    
    # Configure the LLM with structured output
    llm = ModelFactory.get_model(provider="openai",model_name="gpt-4o")
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    
    # Create the chain and execute
    rewriter = rewrite_prompt | structured_llm
    rewritten = rewriter.invoke({})
    
    # Update state with result
    state["question"] = rewritten.question
    state["attempts"] += 1
    log.info(f"Rewritten question: {state['question']}")
    
    return state


#-------------------------------------------------------------------------
# Generate humorous response for irrelevant questions
#-------------------------------------------------------------------------
def generate_funny_response(state: AgentState):
    log.info("Generating a funny response for an unrelated question.")
    
    # Define the system prompt
    system_prompt = """
    You are a charming and funny assistant who responds in a playful manner.
    DO NOT use emojis or any other symbols in your response.
    """.strip()
    
    # Define the human message
    human_message = """
    I can not help with that, but doesn't asking questions make you hungry? 
    You can always order something delicious.
    """.strip()
    
    # Create the prompt template
    funny_prompt = create_chat_prompt(system_prompt, human_message)
    
    # Configure and execute the LLM
    llm = ModelFactory.get_model(provider="openai",model_name="gpt-4o")
    funny_response = funny_prompt | llm | StrOutputParser()
    message = funny_response.invoke({})
    
    # Update state with result
    state["query_result"] = message
    log.info("Generated funny response.")
    
    return state


#-------------------------------------------------------------------------
# Handle maximum iterations reached
#-------------------------------------------------------------------------
def end_max_iterations(state: AgentState):
    state["query_result"] = "Please try again."
    log.info("Maximum attempts reached. Ending the workflow.")
    return state