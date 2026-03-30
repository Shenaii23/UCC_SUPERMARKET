"""
intents/general_chat.py
Handles general conversation and off-topic queries while staying domain-specific.
Lets LLM handle polite closings (goodbye, thanks, etc).
"""
import json
from typing import Dict


class GeneralChatHandler:
    """Handles general chat messages with domain-specific context"""
    
    # Domain-specific topics and redirection suggestions
    DOMAIN_TOPICS = {
        "shopping_help": {
            "keywords": ["help", "how do i", "how can i", "what can you", "can you help"],
            "response_template": "I'd be happy to help! I can assist you with:\n• Finding products (e.g., 'Do you have chicken?')\n• Getting recipes (e.g., 'How do I make pasta?')\n• Checking what you can make with your cart items\n• Building your shopping list\nWhat would you like to do?"
        },
        "store_info": {
            "keywords": ["store", "hours", "location", "phone", "address", "contact"],
            "response_template": "For information about our store hours, location, or contact details, please visit our website or call us directly. In the meantime, I can help you shop for groceries and find recipes! What are you looking for today?"
        }
    }
    
    @staticmethod
    def detect_topic(message: str) -> str:
        """Detect which general chat topic the message belongs to"""
        message_lower = message.lower().strip()
        
        for topic, config in GeneralChatHandler.DOMAIN_TOPICS.items():
            keywords = config.get("keywords", [])
            if keywords and any(keyword in message_lower for keyword in keywords):
                return topic
        
        return "general_query"
    
    @staticmethod
    def has_quick_response(message: str) -> bool:
        """Check if this message has a pre-written response"""
        return GeneralChatHandler.detect_topic(message) != "general_query"
    
    @staticmethod
    def get_quick_response(message: str) -> Dict:
        """Get quick response if available"""
        topic = GeneralChatHandler.detect_topic(message)
        config = GeneralChatHandler.DOMAIN_TOPICS.get(topic)
        
        if not config:
            return None
        
        return {
            "message": config["response_template"],
            "topic": topic,
            "action_ready": False
        }


async def prepare_general_chat_response(user_message: str) -> str:
    """
    Prepare system prompt for general chat questions.
    Lets LLM handle polite closings, off-topic questions, etc.
    Only redirects when necessary.
    """
    
    system_prompt = f"""
    You are a friendly customer service assistant for UCC Supermarket.
    The customer has sent you a message that's not a direct shopping request.
    
    Customer message: "{user_message}"
    
    INSTRUCTIONS:
    1. If it's a polite closing (goodbye, thanks, have a good day, etc):
       - Respond warmly and naturally
       - Wish them well
       - Mention they can ask for help anytime
       - Don't force shopping into it
    
    2. If it's off-topic but friendly (jokes, general chat):
       - Be conversational and warm
       - If relevant, gently mention what you CAN help with
       - Keep it brief and natural
    
    3. If it's a question unrelated to supermarket:
       - Politely explain that's outside your expertise
       - Redirect to supermarket services naturally
       - Don't be pushy
    
    SERVICES YOU CAN HELP WITH:
    - Finding products in inventory
    - Getting recipe ideas and instructions
    - Checking what recipes can be made with cart items
    - Managing shopping cart
    - Answering questions about store policies
    
    Be warm, conversational, and human-like. Don't sound like you're reading from a script.
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Your natural, conversational response",
        "action_ready": false
    }}
    """
    
    return system_prompt


async def handle_general_chat(user_message: str):
    """
    Handle general chat.
    - For quick responses (store info, help requests): return immediately
    - For everything else (goodbye, thanks, off-topic): use LLM for natural response
    """
    from services.user_intent_executive import message_to_llm
    
    # Check if we have a quick response for this
    if GeneralChatHandler.has_quick_response(user_message):
        quick_response = GeneralChatHandler.get_quick_response(user_message)
        return json.dumps(quick_response)
    
    # For everything else (polite closings, off-topic, jokes, etc), use LLM
    system_prompt = await prepare_general_chat_response(user_message)
    llm_response = await message_to_llm(system_prompt, " ", [])
    return llm_response