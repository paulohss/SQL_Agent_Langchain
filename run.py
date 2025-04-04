from workflow import app
from dotenv import load_dotenv
import os
from util.logger import log 

# Load environment variables for API keys
load_dotenv()

def main():
    # Example configuration with a fake user ID
    fake_config = {
        "configurable": {
            "current_user_id": 1  # Assuming user ID 1 exists in database
        }
    }
    
    # Example questions
    examples = [
        "Tell me a joke.",
        "Create a new order for Spaghetti Carbonara.",
        "What foods are available for purchase?",
        "Show me my orders.",
        "What's the price of Pizza Margherita?"
    ]
    
    # Run examples
    for i, question in enumerate(examples):
        log.infoAndPrint(f"\n---> Example {i+1}: '{question}' ---")
        result = app.invoke(
            {"question": question, "attempts": 0}, 
            config=fake_config
        )
        log.infoAndPrint(f"---> Result: {result['query_result']}")

if __name__ == "__main__":
    main()