from typing import Optional, List
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz
import re

def extract_subcategory_from_response(
    user_message: str, 
    available_subcategories: List[str]
) -> Optional[str]:
    """
    Extract subcategory from user's response with fuzzy matching, stemming, and ordinal support.
    
    NOW TAKES: available_subcategories list as parameter (from session state)
    
    Args:
        user_message: The user's input (e.g., "oxtail", "first one", "do you have beef")
        available_subcategories: List of valid subcategory names from session
                                 (e.g., ["Chicken", "Beef", "Oxtail", ...])
    
    Returns:
        str: The matched subcategory name, or None to trigger LLM fallback
    
    Examples:
    - extract_subcategory_from_response("oxtail", ["Chicken", "Beef", "Oxtail"]) → "Oxtail"
    - extract_subcategory_from_response("first one", ["Chicken", "Beef"]) → "Chicken"
    - extract_subcategory_from_response("ostails", ["Chicken", "Beef", "Oxtail"]) → "Oxtail"
    - extract_subcategory_from_response("xyz", ["Chicken", "Beef"]) → None (LLM fallback)
    """
    
    if not available_subcategories:
        print("[WARNING] extract_subcategory_from_response: No subcategories provided")
        return None
    
    stemmer = PorterStemmer()
    user_lower = user_message.lower().strip()
    
    # ========== ORDINAL MATCHING (handles "first one", "1st", "number 1", etc.) ==========
    ordinal_matches = {
        r'\b(first|1st|number\s*1|one)\b': 0,
        r'\b(second|2nd|number\s*2|two)\b': 1,
        r'\b(third|3rd|number\s*3|three)\b': 2,
        r'\b(fourth|4th|number\s*4|four)\b': 3,
        r'\b(fifth|5th|number\s*5|five)\b': 4,
        r'\b(sixth|6th|number\s*6|six)\b': 5,
        r'\b(seventh|7th|number\s*7|seven)\b': 6,
        r'\b(eighth|8th|number\s*8|eight)\b': 7,
        r'\b(ninth|9th|number\s*9|nine)\b': 8,
        r'\b(tenth|10th|number\s*10|ten)\b': 9,
    }
    
    for pattern, index in ordinal_matches.items():
        if re.search(pattern, user_lower):
            if 0 <= index < len(available_subcategories):
                matched = available_subcategories[index]
                print(f"[SUBCATEGORY] Ordinal match: '{user_message}' → {matched}")
                return matched
    
    # ========== EXACT SUBSTRING MATCH (best case) ==========
    for subcategory in available_subcategories:
        if subcategory.lower() in user_lower or user_lower in subcategory.lower():
            print(f"[SUBCATEGORY] Exact match: '{user_message}' → {subcategory}")
            return subcategory
    
    # ========== STEM-BASED MATCHING ==========
    def stem_normalize(word: str) -> str:
        """Normalize word using Porter stemmer"""
        return stemmer.stem(word.lower())
    
    def extract_keywords(text: str) -> list:
        """Extract meaningful keywords from text, excluding stop words"""
        EXCLUDED_WORDS = {
            "the", "a", "an", "some", "any", "you", "have", "please", "show",
            "want", "stock", "inventory", "check", "got", "gotten", "available",
            "our", "what", "which", "do", "you", "have", "is", "i", "me", "my",
            "di", "fuss", "one", "first", "second", "third", "1st", "2nd", "3rd",
            "number", "di fuss"
        }
        
        words = text.split()
        IMPORTANT_SHORT = {"ham", "hot", "pig", "nut", "egg"}
        
        keywords = [
            w for w in words
            if (len(w) > 2 or w.lower() in IMPORTANT_SHORT) and w.lower() not in EXCLUDED_WORDS
        ]
        return keywords
    
    normalized_message_stems = [stem_normalize(kw) for kw in extract_keywords(user_message)]
    
    for subcategory in available_subcategories:
        normalized_subcategory = stem_normalize(subcategory)
        
        # Check if any keyword stem is in the subcategory stem
        for msg_stem in normalized_message_stems:
            if msg_stem in normalized_subcategory or normalized_subcategory in msg_stem:
                print(f"[SUBCATEGORY] Stem match: '{user_message}' → {subcategory}")
                return subcategory
    
    # ========== FUZZY MATCHING ==========
    best_match = None
    best_score = 0
    
    for subcategory in available_subcategories:
        fuzzy_score = fuzz.ratio(user_lower, subcategory.lower())
        partial_score = fuzz.partial_ratio(user_lower, subcategory.lower())
        
        # Combine scores (partial gets 40% weight for typo tolerance)
        combined_score = (fuzzy_score * 0.6) + (partial_score * 0.4)
        
        if combined_score > best_score and combined_score >= 70:  # 70% threshold
            best_score = combined_score
            best_match = subcategory
    
    if best_match:
        print(f"[SUBCATEGORY] Fuzzy match: '{user_message}' → {best_match} (score: {best_score:.1f})")
        return best_match
    
    # ========== NO MATCH FOUND ==========
    print(f"[SUBCATEGORY] No match found for '{user_message}'. Best score was {best_score:.1f}. Falling back to LLM...")
    return None  # Caller will handle LLM fallback


# ========== HELPER: BUILD DYNAMIC SUBCATEGORY MAP FROM INVENTORY ==========
def build_category_subcategory_map(inventory_df) -> dict:
    """
    Build a dynamic mapping from inventory data:
    {
        "Meat & Poultry": ["Chicken", "Beef", "Oxtail", ...],
        "Produce": ["Apple", "Banana", "Carrot", ...],
        ...
    }
    
    Args:
        inventory_df: Pandas DataFrame with columns: category, product_name, etc.
    
    Returns:
        dict: {category: [subcategories]}
    """
    category_map = {}
    
    for _, row in inventory_df.iterrows():
        category = row.get("category")
        product_name = row.get("product_name")
        
        if not category or not product_name:
            continue
        
        # Extract subcategory from product name
        # Strategy: Use first word as default fallback
        words = product_name.split()
        subcategory = words[0] if words else None
        
        # Optional: Keyword-based extraction for better accuracy
        product_lower = product_name.lower()
        keyword_map = {
            "chicken": "Chicken", "beef": "Beef", "pork": "Pork",
            "oxtail": "Oxtail", "fish": "Fish", "shrimp": "Shrimp",
            "milk": "Milk", "cheese": "Cheese", "bread": "Bread",
            "rice": "Rice", "juice": "Juice", "water": "Water",
            "turkey": "Turkey", "lamb": "Lamb", "duck": "Duck",
            "apple": "Apple", "banana": "Banana", "carrot": "Carrot",
            "tomato": "Tomato", "lettuce": "Lettuce", "broccoli": "Broccoli",
        }
        
        for keyword, mapped_name in keyword_map.items():
            if keyword in product_lower:
                subcategory = mapped_name
                break
        
        if not subcategory:
            continue
        
        # Add to map
        if category not in category_map:
            category_map[category] = set()
        category_map[category].add(subcategory)
    
    # Convert sets → sorted lists
    return {
        cat: sorted(list(subs))
        for cat, subs in category_map.items()
    }


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    # Example 1: Using extracted subcategories from state
    print("=" * 80)
    print("EXAMPLE 1: Using last_subcategories from session state")
    print("=" * 80)
    
    # Simulating state from session
    last_subcategories = [
        {"subcategory": "Chicken"},
        {"subcategory": "Beef"},
        {"subcategory": "Pork"},
        {"subcategory": "Oxtail"},
    ]
    
    # Extract just the names
    available_subs = [s["subcategory"] for s in last_subcategories]
    
    test_cases = [
        "oxtail",
        "oxtai",
        "first one",
        "2nd",
        "beef",
        "do you have pork",
        "xyz123",
    ]
    
    for test in test_cases:
        result = extract_subcategory_from_response(test, available_subs)
        status = "✅" if result else "❌"
        print(f"{status} '{test}' → {result}\n")
    
    # Example 2: Dynamic mapping from inventory
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Building category map from inventory (Pandas)")
    print("=" * 80)
    print("""
    # This would be in your code:
    category_map = build_category_subcategory_map(inventory_df)
    
    # Then when user selects a category, get available subs:
    available_subcategories = category_map.get("Meat & Poultry", [])
    
    # Use it:
    result = extract_subcategory_from_response(user_message, available_subcategories)
    """)