from pydantic import BaseModel, Field
from typing import List, Optional

# a class to check intent 
class IntentQuery(BaseModel):
    """ A user query to check intent for LLM processing. 
    The message is the raw user input, and the LLM will determine if it contains an actionable intent
      (like adding to cart, stock check, cart management, product search, or if it's just a general message.)
      This will use ollama with a system prompt that defines the intents and how to detect them. 
      The LLM will return a structured response indicating the detected intent(s).
      Example input:
    {
        "message": "Add 2 rice to my cart"
    }"""

    message: str = Field(..., description="The raw user message to be processed by the LLM")
    user_id: str 
    cart: list = []


# Another class for entity extration
class Entity(BaseModel):
    """ A user query to extract entities for LLM processing. 
    The message is the raw user input, and the LLM will extract relevant entities like product names, quantities, etc. 
    This can be used after intent detection to get the details needed to perform the action."""
    product_name: Optional[str] = Field(None, description="The name of the product mentioned in the user message")
    quantity: Optional[int] = Field(None, description="The quantity of the product mentioned in the user message")
    recipe_name: Optional[str] = Field(None, description="The name of the recipe mentioned in the user message")
    servings: Optional[int] = Field(None, description="The number of servings mentioned in the user message")
    product_codes: Optional[List[str]] = Field(None, description="A list of product codes mentioned in the user message")
    budget: Optional[float] = Field(None, description="The budget mentioned in the user message")
    category_hint: Optional[str] = Field(None, description="The logical category (e.g., Produce, Dairy, Canned Goods)")


# A function to do execute the users intent
class IntentExecution(BaseModel):
    """ A user query to execute an intent for LLM processing. This class will call the specific
      function to perform the action determined by the intent detection."""
    intent: str = Field(..., description="The detected intent to execute")
    entities: Entity = Field(..., description="The extracted entities needed to perform the action")

    
class LLMResponse(BaseModel):
    """ A response class to standardize the output from the LLM. It contains a response
      message and an optional action that can be taken by the frontend."""
    response: str = Field(..., description="The response message from the LLM")
    action: Optional[dict] = Field(None, description="An optional action that can be taken by the frontend")