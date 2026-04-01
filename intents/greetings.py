"""This will handle all general chat intents."""

from models.llm_classes import Entity, IntentExecution
import re
import random

def normalize(name: str) -> str:
    name = re.sub(r'[^\w\s]', '', name)
    return name.lower().replace("-", " ").strip()
 
# Greeting lists and responses
LIST_OF_GREETINGS = [
    # Standard English
    'hi', 'hello', 'hey', 'hey there', 'howdy', 'yo', 'sup', 'whats up', "what's up",
    'good morning', 'good afternoon', 'good evening', 'good day', 'greetings',
    # Casual / slang English
    'yo bro', 'yo man', 'hey bro', 'hey man', 'sup bro', 'sup man',
    'wagwan', 'wagwan bro', 'wagwan man', 'wagwan mi boss',
    'what up', 'whats good', "what's good", 'what going on', "what's going on",
    # Jamaican Patois core greetings
    'wah gwaan', 'wah gwaan', 'wahgwaan', 'wa gwaan', 'wagwaan',
    'wah gwan', 'wa gwan', 'wagwan', 'wah gwaan mi boss',
    'wah gwaan mi friend', 'wah gwaan mi fren',
    # Patois variations / phonetic spellings
    'waah gwaan', 'wha gwaan', 'whagwaan', 'wagwaan',
    'wah gwarn', 'wagwarn', 'wagwarn', 'wa gwan',
    # Time-based Patois greetings
    'mornin', 'mawnin', 'good mawnin', 'gud mawnin',
    'evenin', 'good evenin', 'gud evenin',
    'good aftanoon', 'aftanoon',
    # Friendly / expressive Jamaican greetings
    'bless', 'bless up', 'blessings', 'respect', 'respek',
    'big up', 'big up yuhself', 'big up yourself',
    'hail', 'hail up', 'hail mi boss', 'heyo', 'heya',
    # Social / checking in
    'how yuh stay', 'how yuh deh', 'how yuh dey',
    'how yuh do', 'how yuh a do',
    'how things', 'how tings', 'how tings stay',
    'everything good', 'everyting good', 'erryting good',
    # Short forms / lazy typing
    'hii', 'heyy', 'helloo', 'yoo', 'sup', 'supz',
    'gm', 'gud mrn', 'gud mornin', 'gn', 'ge',
    # Mixed English + Patois
    'hey wagwan', 'yo wagwan', 'hello wagwan',
    'hi wagwan', 'wagwan hello', 'hallo', 'hallo there', 'helio',
    # Extra edge casuals
    'what a gwaan', 'wha a gwaan', 'wa a gwaan',
    'wagwan deh', 'wagwan deh suh',
    'wah gwaan deh so',
    # Question-style greetings
    'you good', 'u good', 'you alright', 'yuh good',
    'yuh alright', 'everything irie', 'irie', "yow", "whats up", 'what can u do',
    'what can you do', 'what do you do', 'how can you help', 'how can you help me',
    'what can you help me with', 'what services do you offer',
    'can you help me shop', 'can you help me find products', 'can you recommend items', 'can you suggest recipes', 'can you help me with groceries',
]

 
GREETING_RESPONSES = [
    "Hi! I'm UCC Supermarket Assistant. I can help you find recipes, check what's in stock, and add items to your cart. What can I help you with today?",
    "Hello! I'm your UCC Supermarket Assistant. I can suggest meals, check ingredient availability, or help you shop. What are you looking for?",
    "Hey! I'm UCC Supermarket Assistant. I can help you discover recipes, check our inventory, or build your cart. How can I assist you today?",
    "Hi there! UCC Supermarket Assistant here. I can help with recipes, ingredient checks, and shopping. What do you need today?",
    "Welcome! I'm UCC Supermarket Assistant. I can help you find meals to cook, see what's available, and add items to your cart. What can I do for you?",
    "Hello! This is UCC Supermarket Assistant. I can help you plan meals, check ingredients, and shop easily. How can I help?",
    "Hey there! I'm UCC Supermarket Assistant. Need help with recipes, checking stock, or adding items to your cart?",
    "Hi! UCC Supermarket Assistant here. I can help you find what to cook, check if we have ingredients, and manage your shopping list.",
    "Hello! I'm your UCC Supermarket Assistant. I can help with meal ideas, stock checks, and adding items to your cart. What would you like to do?",
    "Hi! I'm UCC Supermarket Assistant. Ask me about recipes, ingredients, or shopping and I'll take care of it for you."
]

AFFECTIONAL_GREETINGS = [
    # English variations
    "you good", "u good", "you alright", "yuh good", "yuh alright",
    "everything irie", "irie", "hru", "how are you", "how are you doing",
    "how are you doing?", "how you doing", "how you doing?", "how you doin",
    "how you doin?", "how u doin", "how u doing",

    # Casual / slang
    "wassup", "what up", "whats good", "what's good", "what going on",
    "what's going on", "sup", "sup bro", "sup man", 

    # Jamaican Patois affectionate
    "how tings", "how tings deh", "how tings stay", "everyting good",
    "how yuh stay", "how yuh deh", "how yuh dey", "how yuh do", "how yuh a do",
    "irie vibes", "bless up", "big up"
]

AFFECTIONAL_GREETING_RESPONSES = [
    "I'm doing great, thanks for asking! I'm UCC Supermarket Assistant — I can help you find recipes, check ingredient availability, or add items to your cart.",
    "All good here! I’m your UCC Supermarket Assistant. I can help you plan meals, see what’s in stock, or shop online easily.",
    "Doing fine, thank you! I’m UCC Supermarket Assistant — you can ask me about recipes, ingredients, or add products to your cart.",
    "I’m good, hope you are too! I can help you discover recipes, check our shelves, and build your shopping list.",
    "All irie! I’m UCC Supermarket Assistant. Ask me for meal ideas, check if we have ingredients, or manage your cart.",
    "Doing well, thanks! I can help you find meals to cook, see what ingredients are available, and add items to your cart.",
    "I’m fine, thanks for checking! I’m here to help you with recipes, stock checks, or adding products to your shopping cart.",
    "Feeling good! I’m UCC Supermarket Assistant — I can suggest meals, tell you what we have in store, or help you shop smarter.",
    "Doing great! I can help you plan meals, check ingredient availability, and quickly add items to your cart.",
    "I’m good, thanks! You can ask me things like 'what can I cook with chicken', 'do you have ingredients for curry', or 'suggest a meal under $2000'.",
]
 
PURPOSE_MESSAGES = [
    "what can you do",
    "what do you do",
    "how can you help",
    "how can you help me",
    "what can you help me with",
    "Can u order for me",
    "what services do you offer",

    "can you help me shop",
    "can you help me find products",
    "can you recommend items",
    "can you suggest recipes",
    "can you help me with groceries",

    "help",
    "help me",
]

# Messages users might type to check if the assistant/system is online
STATUS_CHECK_MESSAGES = [
    "are you online",
    "are you there",
    "are you on",
    "is it on",
    "are you working",
    "are you active",
    "can you hear me",
    "you there",
    "are you awake",
    "working?",
    "online?",
    "can you respond",
    "hi, are you there",
    "hello, are you online",
    "u workin",
    "yah work",
    "yuh work",
    "yuh there",
    "yuh online",
    "yuh active",
    "yuh can hear me",
    "yuh awake",
    "yuh working",
    'wake up',
    'are you up',
    'you up',
    'you awake',
    'you alive',
    'wake',
    'status check',
    'system check',
    'are you functioning',
    'are you operational',
    'are you responsive',
    'can you respond',
    'ping',
    'yuh down',
    'online',
    'up',
    'awake', 'active', 'working', 'responding', 'status', 'check', 'system', 'functioning', 'operational', 'responsive',
    'wake up', 'grand rising'
]

# Static responses for these messages
STATUS_CHECK_RESPONSES = [
    "Yes, I’m online and ready to help you!",
    "I’m here! How can I assist you today?",
    "All systems go! What would you like help with?",
    "I’m active and ready to assist.",
    "Yes! I can help you with products, recipes, or your cart.",
    "I’m here and listening. How can I help?",
    "Everything’s working fine! What do you need today?",
]

FORMAL_GREETINGS = [
    "good morning", "good afternoon", "good evening", "good day", "greetings", 
    'salutations', 'pleased to meet you', 'how do you do', 'it’s a pleasure to meet you',
    'nice to meet you', 'delighted to meet you', 'how are you doing', 'how are you doing?', 'how you doing', 'how you doing?', 'how doin', 'how doin?', 'how u doin', 'how u doing'
    ]

FORMAL_RESPONSES = [
    "Good day! I'm UCC Supermarket Assistant. I can help you find recipes, check what's in stock, and add items to your cart. What can I help you with today?",
    "Hi! I'm UCC Supermarket Assistant. I can help you find products, check what's in stock, and add items to your cart. What can I help you with today?",
    "Hello! I'm UCC Supermarket Assistant. I can help you find products, check what's in stock, and add items to your cart. What can I help you with today?",   
    "Greetings! I'm UCC Supermarket Assistant. I can help you find recipes, check what's in stock, and add items to your cart. What can I assist you with today?",
    "I hope you're having a wonderful day! I'm UCC Supermarket Assistant. I can help you find recipes, check what's in stock, and add items to your cart. What can I do for you today?",
    "Hi there! I'm UCC Supermarket Assistant. I can help you find products, check what's in stock, and add items to your cart. What can I help you with today?",
    "Hi! I'm UCC Supermarket Assistant. I can help you find products, check what's in stock, and add items to your cart. What can I help you with today?",]

GOODBYE_MESSAGES = [
    "bye", "goodbye", "see you later", "talk to you later", "catch you later", "farewell", "take care", "have a good day", "later", "peace out", "cya", "see ya", "adios", "sayonara", "cheerio",
    "i'm out", "i'm off", "got to go", "need to go", "leaving now", "heading out", "catch you on the flip side", "until next time", "have a good one", "stay safe", "take it easy", "see you soon", "talk soon", "catch you later", "good night", "sweet dreams"
]

GOODBYE_RESPONSES = [
    "Goodbye! I hope I was able to help you. Come back anytime if you have more questions or need assistance with recipes, stock, or your cart!",
    "See you later! If you need help with recipes, checking stock, or your cart, just ask for me again. Have a great day!",
    "Have a good day! If you need help with recipes, checking stock, or your cart, just ask for me again. See you later!",
    "Farewell! If you need help with recipes, checking stock, or your cart, just ask for me again. Take care!",
    "Take care! If you need help with recipes, checking stock, or your cart, just ask for me again. See you later!",
    "Goodbye! If you need help with recipes, checking stock, or your cart, just ask for me again. Have a great day!",
    "See you later! If you need help with recipes, checking stock, or your cart, just ask for me again. Take care!"]

PURPOSE_RESPONSES = [
    "Hi! I’m your UCC Supermarket assistant. I can help you find products, check what’s in stock, suggest recipes, and add items to your cart.",
    
    "I can help you shop by finding products, recommending meals, checking availability, and adding items to your cart.",
    
    "I’m here to help with groceries! I can suggest recipes, check what we have in stock, and help you choose what to buy.",
    
    "You can ask me to find products, suggest meals, check prices, or help you build your cart.",
    
    "Need help shopping? I can recommend items, suggest recipes, and check what’s available in store.",
    
    "I can help you quickly find groceries, suggest what to cook, and add items to your cart.",
    
    "I’m here to make shopping easier. I can help you find items, suggest recipes, and check availability.",
    
    "Hi, you can ask me about products, recipes, or what we have in stock—I’ve got you covered!",
]

CART_MESSAGES = [
    "whats in my cart",
    "what's in my cart",
    "show my cart",
    "view my cart",
    "my cart",
    "check my cart",
]

def handle_cart(user_id):
    """Handle user's cart"""
    from services.context_handler import get_user_state
    # Get cartfrom the json file using the user_id, if it doesn't exist return empty cart
    from intents.cart_logic import cart_summary

    cart = cart_summary(user_id)
    print("[Cart Handler] Current cart for user_id", user_id, ":", cart)

    if not cart:
        return "Your cart is empty right now. Would you like to add some items? I can help you find products and add them to your cart!"
    return f"You have the following items in your cart: {cart}."

def is_greeting(user_message: str) -> str | None:
    """
    Check if message is a greeting.
    
    Args:
        user_message: The user's message
        
    Returns:
        A random greeting response if matched, None otherwise
    """
    cleaned_message = normalize(user_message)
    for greeting in LIST_OF_GREETINGS:
        if cleaned_message == greeting:
            return random.choice(GREETING_RESPONSES)
        elif cleaned_message in AFFECTIONAL_GREETINGS:
            return random.choice(AFFECTIONAL_GREETING_RESPONSES)
        elif cleaned_message in PURPOSE_MESSAGES:
            return random.choice(PURPOSE_RESPONSES)
        elif cleaned_message in STATUS_CHECK_MESSAGES:
            return random.choice(STATUS_CHECK_RESPONSES)
        elif cleaned_message in FORMAL_GREETINGS:
            return random.choice(FORMAL_RESPONSES)
        elif cleaned_message in GOODBYE_MESSAGES:
            return random.choice(GOODBYE_RESPONSES)
    return None