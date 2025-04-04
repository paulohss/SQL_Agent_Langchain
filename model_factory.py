from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpoint

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
                "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
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
        else:
            # Default to OpenAI
            return ChatOpenAI(temperature=temperature, model="gpt-4o")