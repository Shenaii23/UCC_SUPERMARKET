"""
intents/terms_and_conditions.py
Handles the "terms_and_conditions" intent by retrieving relevant policy documents
"""
import json
from documents.document_handler import get_document_context



async def prepare_terms_and_conditions_response(user_message: str) -> str:
    """
    Prepare a system prompt for the LLM that includes relevant policy documents.
    The LLM will answer the user's question based on the actual policy text.
    """
    
    # Get relevant sections from policy documents
    document_context = get_document_context(user_message)
    
    system_prompt = f"""
    You are a helpful customer service assistant for UCC Supermarket.
    A customer is asking about our store policies, terms, or privacy practices.
    
    Below are the relevant sections from our official policy documents:
    
    {document_context}
    
    TASK:
    Answer the customer's question based ONLY on the information provided above.
    - Be clear, friendly, and professional
    - If the information isn't in the documents, say so
    - Don't make up policies or information
    - Provide specific details like timeframes, procedures, etc.
    - If there are multiple related points, organize them clearly
    
    User's question: "{user_message}"
    
    Provide a helpful, accurate answer based on the policy documents above.
    Format your response as JSON:
    {{
        "message": "Your helpful response here",
        "source": "UCC Supermarket Policies"
    }}
    """
    
    return system_prompt


async def terms_and_conditions(user_message: str):
    """
    Handle the terms_and_conditions intent.
    Retrieve policy documents and generate an LLM response.
    """
    from services.user_intent_executive import message_to_llm
    
    # Get the system prompt with document context
    system_prompt = await prepare_terms_and_conditions_response(user_message)
    
    # Call LLM to generate answer based on documents
    llm_response = await message_to_llm(system_prompt, " ", [])
    
    
    return llm_response