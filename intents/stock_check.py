from datasets.data import inventory_df, sales_df
from services.context_handler import get_user_state
from typing import Optional, List, Dict, Any
from rapidfuzz import process, fuzz


import re
import random


def normalize(name: str) -> str:
    # Remove any 
    name = re.sub(r'[^\w\s]', '', name)

    return name.lower().replace("-", " ").strip()


def rank_results(results: list, top_n: int = 3, boost_n: int = 2) -> list:
    """
    Merges sales data into results, ranks by units_sold.
    Returns top_n best sellers + boost_n slow movers (deduplicated).
    """
    # Join sales data onto results by product_code
    sales_lookup = sales_df.set_index("product_code")["units_sold"].to_dict()

    for r in results:
        r["units_sold"] = sales_lookup.get(r["product_code"], 0)

    sorted_results = sorted(results, key=lambda x: x["units_sold"], reverse=True)

    top_sellers = sorted_results[:top_n]
    slow_movers = sorted_results[-boost_n:]

    # Deduplicate
    seen    = {r["product_code"] for r in top_sellers}
    boosted = [r for r in slow_movers if r["product_code"] not in seen]

    final = top_sellers + boosted

    # Clean up units_sold before returning — frontend doesn't need it
    for r in final:
        r.pop("units_sold", None)

    return final


def get_inventory_data(search_term, category_hint: str = None) -> list:
    if not search_term or len(search_term) < 2:
        return []

    search_lower   = search_term.lower().strip()
    inventory_names = inventory_df["product_name"].tolist()

    # Pass 1: exact word / prefix match
    exact_matches = [
        name for name in inventory_names
        if search_lower in normalize(name).split()
        or normalize(name).startswith(search_lower)
    ]

    # Pass 2: fuzzy fallback
    if not exact_matches:
        normalized_inventory = [normalize(n) for n in inventory_names]
        raw = process.extract(normalize(search_term), normalized_inventory, scorer=fuzz.WRatio)
        matched_indices = [
            idx for _, score, idx in raw
            if score >= 88 and normalize(search_term) in normalize(inventory_names[idx])
        ]
        exact_matches = [inventory_names[i] for i in matched_indices]

    if not exact_matches:
        return []

    matches_df = inventory_df[inventory_df["product_name"].isin(exact_matches)]

    # Filter by category if provided
    if category_hint:
        cat_filtered = matches_df[matches_df["category"].str.lower() == category_hint.lower()]
        if not cat_filtered.empty:
            matches_df = cat_filtered

    results = matches_df[["product_code", "product_name", "quantity", "price"]].to_dict(orient="records")

    # Rank: top sellers + slow movers boost
    return rank_results(results)


# Was the follow up a stock check
def was_stock_check(user_message: str, user_id: str) -> Optional[str]:
    # Get the last intent from the user session
   
    # If the last intent was stock check and the current message is in a list of 
    # follow up keywords like yes, no,
    STANDALONE_YES_NO_RESPONSES = [
    "yes", "yep", "yeah", "yup", "sure", "ok", "okay", "alright", "affirmative",
    "no", "nope", "nah", "negative", "not really", "never",
    "got it", "alrighty", "sounds good", "cool",
    "thanks", "thank you", "roger", "understood", "right", 'hmm', 'hmmm'
    ]

    STANDALONE_YES_NO_RESPONSES_MESSAGES = [
        "I’m not sure which one you mean. Could you tell me which product you want?",
            "Oops! Could you clarify which item you’re referring to?",
        "I got your yes/no, but I need to know which product to add. Which one would you like?",
        "Thanks! Can you specify which one from the list you want?",
        "Just to be clear, which product should I go with?",
        "I heard you, but could you confirm the item you mean?",
        "Almost there! Which of the items would you like me to add?",
        "Got it, but I need to know which product you’re choosing. Can you specify?",
        "Thanks for your response! Please tell me which product you’re referring to.",
        "Okay! Could you clarify which item from the list you want me to add?",
    ]

    # Check if the last intent was stock check
    user_state = get_user_state(user_id)
    last_intent = user_state["last_intent"]
    if last_intent == "stock_check":
        # If there is an active pending product selection, do not intercept yes/no here.
        if user_state.get("pending_selection"):
            return None

        # Check if the current message is in the list of standalone yes/no responses
        for i in STANDALONE_YES_NO_RESPONSES:
            if user_message.lower() == i:
                # Get the llm last 
                last_response = user_state.get("last_response", "")
                # Add it to the response to the user
                return random.choice(STANDALONE_YES_NO_RESPONSES_MESSAGES) + "\n\n" + last_response
        return None
    # If the last intent was not stock check or the current message is not in the list of standalone yes/no responses, return None
    return None




