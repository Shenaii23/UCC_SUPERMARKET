import re
from models.llm_classes import IntentExecution, Entity


# ─────────────────────────────────────────────
#  KEYWORD MAPS
# ─────────────────────────────────────────────

INTENT_PATTERNS = [
    # Budget first — most specific, wins over recommend/stock
    ("budget_recipe_suggestion", [
        r"\$\d+", r"\bjmd\b", r"budget", r"under \d+", r"for \d+ dollars",
        r"spend", r"afford", r"cheap", r"cost.{0,10}meal", r"meal.{0,10}cost"
    ]),

    # Cart actions
    ("add_to_cart", [
        r"\badd\b", r"\bput\b.{0,10}\bcart\b", r"\bbasket\b.{0,10}\badd\b",
        r"\bi (want|need|take)\b.{0,20}\bcart\b", r"\bthrow.{0,10}(in|into)\b"
    ]),
    ("remove_from_cart", [
        r"\bremove\b", r"\btake out\b", r"\bdelete\b.{0,10}\bcart\b",
        r"\bdrop\b.{0,10}\bcart\b"
    ]),

    # Recipe availability check
    ("check_recipe_availability", [
        r"can i make", r"do you have.{0,20}(ingredients|stuff).{0,20}for",
        r"have (everything|what i need) (for|to make)",
        r"ingredients.{0,10}(available|in stock)"
    ]),

    # Recommend recipe
    ("recommend_recipe", [
        r"what can i make (with|using)", r"any (meal|dinner|lunch|breakfast|recipe) ideas",
        r"suggest.{0,10}recipe", r"recommend.{0,10}(meal|dish|recipe)",
        r"what (should|could) i (cook|make|prepare)"
    ]),

    # Get recipe
    ("get_recipe", [
        r"how (do i|to) (make|cook|prepare|bake)",
        r"recipe for", r"(give|show) me.{0,10}recipe",
        r"ingredients (for|to make)", r"steps (to|for) (make|cook|bake|prepare)",
        r"what do i need (for|to make)"
    ]),

    # Stock check — availability / price / natural purchase phrases
    ("stock_check", [
        r"\bdo you have\b", r"\bhave\b.{0,10}\?",
        r"\bin stock\b", r"\bavailable\b",
        r"\bhow much (is|does|costs?)\b", r"\bprice of\b", r"\bcost of\b",
        r"\bgot any\b", r"\byou have\b", r"\bany \w+ (left|available)\b",
        r"\bsell\b", r"\bcarry\b",
        r"\bi need\b", r"\bi want\b", r"\bget me\b", r"\bcan i get\b",
        r"\blooking for\b", r"\bwhere.{0,10}(find|get)\b",
        r"\bdo you sell\b", r"\bdo you carry\b",
    ]),

    # General chat — greetings / vague
    ("general_chat", [
        r"^(hi|hello|hey|howdy|yo|sup|what'?s up|good (morning|afternoon|evening))",
        r"how are you", r"can you help", r"who are you", r"what (can|do) you do"
    ]),
]


CATEGORY_HINTS = {
    "Produce":             ["apple", "banana", "mango", "lettuce", "tomato", "carrot", "onion",
                            "potato", "cabbage", "pepper", "cucumber", "spinach", "corn",
                            "pumpkin", "scotch bonnet", "cho cho", "callaloo"],
    "Meat & Poultry":      ["chicken", "beef", "pork", "turkey", "lamb", "goat", "oxtail",
                            "patty", "sausage", "bacon"],
    "Seafood":             ["fish", "shrimp", "lobster", "crab", "snapper", "salmon", "tuna",
                            "mackerel", "sardine"],
    "Dairy & Eggs":        ["milk", "egg", "cheese", "butter", "yogurt", "cream"],
    "Beverages":           ["juice", "water", "soda", "drink", "tea", "coffee", "cola",
                            "lemonade", "rum", "beer"],
    "Bakery":              ["bread", "bun", "cake", "roll", "biscuit", "pastry", "dough"],
    "Grains & Rice":       ["rice", "flour", "oats", "cornmeal", "pasta", "noodle", "wheat"],
    "Canned & Jarred":     ["ackee", "canned", "jarred", "tin", "baked beans", "coconut milk"],
    "Snacks":              ["chips", "crackers", "popcorn", "nuts", "granola bar"],
    "Condiments & Sauces": ["ketchup", "sauce", "mustard", "mayo", "vinegar", "soy sauce",
                            "hot sauce", "dressing"],
    "Spices & Seasonings": ["salt", "pepper", "curry", "thyme", "garlic", "ginger", "seasoning",
                            "allspice", "cinnamon", "nutmeg"],
    "Frozen Foods":        ["frozen", "ice cream"],
    "Baking Supplies":     ["baking powder", "baking soda", "vanilla", "yeast", "cocoa"],
}


# ─────────────────────────────────────────────
#  ENTITY EXTRACTION
# ─────────────────────────────────────────────

def extract_quantity(text: str) -> int | None:
    match = re.search(r"\b(\d+)\s*(kg|lbs?|g|oz|pack|bag|box|carton|bottle|piece|unit|item)?\b", text)
    if match:
        val = int(match.group(1))
        if val < 1000:
            return val
    return None


def extract_budget(text: str) -> float | None:
    match = re.search(r"[\$j]?\s*(\d[\d,]*)\s*(jmd|dollars?|usd)?", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def extract_product_name(text: str) -> str | None:
    cleaned = re.sub(
        r"\b(do you have|got any|you have|is there|any|some|the|a|an|please|get me|"
        r"add|remove|take out|put|i (want|need|would like)|how much (is|does)|price of|"
        r"cost of|in stock|available)\b",
        "", text, flags=re.IGNORECASE
    ).strip(" ?,!")
    words = [w for w in cleaned.split() if len(w) > 1]
    return " ".join(words[:4]) if words else None


def extract_category_hint(product_name: str | None) -> str | None:
    if not product_name:
        return None
    lower = product_name.lower()
    for category, keywords in CATEGORY_HINTS.items():
        if any(kw in lower for kw in keywords):
            return category
    return None


def extract_recipe_name(text: str) -> str | None:
    patterns = [
        r"how (do i|to) (?:make|cook|prepare|bake) (.+?)(?:\?|$)",
        r"recipe for (.+?)(?:\?|$)",
        r"(?:make|cook|prepare|bake) (.+?)(?:\?|$)",
        r"ingredients (?:for|to make) (.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(match.lastindex).strip(" ?,!")
    return None


# ─────────────────────────────────────────────
#  MAIN CLASSIFIER
# ─────────────────────────────────────────────

def classify_intent(user_message: str) -> tuple[IntentExecution, bool]:
    """
    Classifies user intent using keyword/regex rules.
    Fast (~1ms). No LLM call needed.

    Returns:
        (IntentExecution, confident: bool)
        confident=False means no pattern matched — caller should fall back to LLM.
    """
    text = user_message.lower().strip()
    detected_intent = None
    confident = False

    # Walk patterns in priority order — first match wins
    for intent_name, patterns in INTENT_PATTERNS:
        if any(re.search(p, text) for p in patterns):
            detected_intent = intent_name
            confident = True
            break

    # No pattern matched at all — signal the caller to use LLM fallback
    if not confident:
        detected_intent = "general_chat"

    # Extract entities
    product_name  = extract_product_name(text)
    recipe_name   = extract_recipe_name(text)
    quantity      = extract_quantity(text)
    budget        = extract_budget(text) if detected_intent == "budget_recipe_suggestion" else None
    category_hint = extract_category_hint(product_name)

    if detected_intent in ("get_recipe", "check_recipe_availability", "recommend_recipe", "budget_recipe_suggestion"):
        product_name = None

    entities = Entity(
        product_name=product_name,
        quantity=quantity,
        recipe_name=recipe_name,
        budget=budget,
        category_hint=category_hint,
    )

    #print(f"[Classifier] Intent: {detected_intent} | Confident: {confident} | Entities: {entities}")
    return IntentExecution(intent=detected_intent, entities=entities), confident