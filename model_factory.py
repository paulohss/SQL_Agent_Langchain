from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_community.llms import Replicate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import Type, Any, Optional
from pydantic import BaseModel
from util.logger import log


class ModelFactory:
    """Factory for creating different language model instances based on configuration"""
    
    @staticmethod
    def get_model(provider="openai", model_name=None, temperature=0):
        """
        Create and return a language model based on the specified provider
        
        Args:
            provider: The model provider (openai, anthropic, groq, mistral, huggingface)
            model_name: Specific model to use (if None, uses default for provider)
            temperature: Sampling temperature
        
        Returns:
            Configured language model instance
        """
        # Set default model names if not specified
        if model_name is None:
            model_name = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-haiku-20240307",
                "groq": "llama3-70b-8192",
                "mistral": "mistral-large-latest",
                "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
                "llama_ollama": "llama3.1:8b-instruct-q4_0"
            }.get(provider, "gpt-4o")
        
        # Create the appropriate model instance
        if provider == "openai":
            return ChatOpenAI(temperature=temperature, model=model_name)
        
        elif provider == "anthropic":
            return ChatAnthropic(temperature=temperature, model=model_name)
        
        elif provider == "groq":
            return ChatGroq(temperature=temperature, model=model_name)
        
        elif provider == "mistral":
            return ChatMistralAI(temperature=temperature, model=model_name)
        
        elif provider == "huggingface":
            return HuggingFaceEndpoint(temperature=temperature, repo_id=model_name)
        
        elif provider == "llama_replicate":
                # Use Replicate API for Llama 3
                return Replicate(model=model_name,temperature=temperature,input={"temperature": temperature, "max_length": 2048}
                                 )        
        elif provider == "llama_ollama":
                return ChatOllama(model=model_name,temperature=temperature)
            
        else:
            # Default to OpenAI
            log.warning(f"Unknown provider '{provider}', defaulting to OpenAI")
            return ChatOpenAI(temperature=temperature, model="gpt-4o")
    
    
    @staticmethod
    def with_structured_output(llm, pydantic_model: Type[BaseModel]):
        """
        Apply structured output capability to an LLM, with fallback for unsupported models
        
        Args:
            llm: The language model instance
            pydantic_model: The Pydantic model to use for structured output
            
        Returns:
            A runnable chain that produces structured output
        """
        # First try the native method if it exists
        try:
            # Check if the model has native structured output support
            if hasattr(llm, "with_structured_output"):
                log.info(f"Using native structured output for {type(llm).__name__}")
                return llm.with_structured_output(pydantic_model)
        except Exception as e:
            log.warning(f"Native structured output failed: {str(e)}")
            
        # Fall back to manual implementation
        log.info(f"Using custom structured output for {type(llm).__name__}")
        
        # Create a parser for the Pydantic model
        parser = PydanticOutputParser(pydantic_object=pydantic_model)
        
        # Create format instructions for the model
        format_instructions = f"""
        Your response must be formatted as a JSON object that conforms to this schema:
        {pydantic_model.model_json_schema()}
        
        Ensure your response is valid JSON. Do not include explanations before or after the JSON.
        """
        
        
        # Function to process inputs and handle both prompt objects and dictionaries
        def process_input(input_data):
            if isinstance(input_data, dict):
                # If it's already a dict, use it directly
                prompt = input_data
            else:
                # If it's a prompt or string, use it as is
                prompt = input_data
                
            # Add format instructions to the prompt
            if hasattr(prompt, "messages"):
                try:
                    # For chat prompts
                    from langchain_core.messages import SystemMessage, HumanMessage
                    
                    messages = list(prompt.messages)
                    system_message_found = False
                    
                    # Check for existing system message
                    for i, message in enumerate(messages):
                        if isinstance(message, SystemMessage):
                            # Update existing system message
                            new_content = message.content + "\n\n" + format_instructions
                            messages[i] = SystemMessage(content=new_content)
                            system_message_found = True
                            break
                    
                    # Add new system message if none exists
                    if not system_message_found:
                        messages.insert(0, SystemMessage(content=format_instructions))
                    
                    # Create updated prompt
                    return prompt.update(messages=messages)
                except Exception as e:
                    log.error(f"Error processing chat messages: {str(e)}")
                    # Fallback to adding format instructions to the first message
                    try:
                        return prompt + "\n\n" + format_instructions
                    except:
                        # If all else fails, just use the format instructions
                        return format_instructions
            else:
                # For string prompts
                return str(prompt) + "\n\n" + format_instructions
        
        
        
        # Function to parse the response into the Pydantic model
        def parse_response(response):
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
                
            try:
                return parser.parse(content)
            except Exception as e:
                log.error(f"Error parsing response: {str(e)}")
                log.error(f"Response content: {content}")
                # Create an empty model instance with default values
                return pydantic_model.model_construct()
        
        # Build a simpler chain that accepts any input format
        custom_chain = (
            RunnableLambda(process_input)
            | llm
            | RunnableLambda(parse_response)
        )
        
        return custom_chain