"""
intents/terms_and_conditions.py
Handles the "terms_and_conditions" intent by retrieving relevant policy documents
"""
import json
from documents.document_handler import get_document_context



def _build_rag_prompt(user_message: str, document_context: str, topic_description: str) -> str:
    """Helper to build a unified, high-quality RAG prompt."""
    return f"""
    You are a helpful customer service assistant for UCC Supermarket.
    A customer is asking {topic_description}.
    
    Below are the relevant sections from our official store guides and policy documents:
    
    {document_context}
    
    TASK:
    Answer the customer's question based on the official documentation provided above.
    - Be clear, friendly, and professional.
    - PRIORITIZE the information in the documents. 
    - If the specific detail (like a rare city name) isn't in the documents, state that it's not listed and suggest contacting the store directly.
    - Provide specific details like hours, phone numbers, or delivery partners (e.g., 7krave, Swift) if they are in the context.
    - Organize multiple points using bullet points for readability.
    
    General Store Contact for Gaps:
    - Email: info@uccsupermarket.com
    - Phone numbers are available in the store location list above.
    
    User's question: "{user_message}"
    
    Provide a helpful, accurate answer.
    Format your response as JSON:
    {{
        "message": "Your helpful response here",
        "source": "UCC Supermarket Official Guides"
    }}
    """


async def terms_and_conditions(user_message: str):
    """Handle the terms_and_conditions intent (wrapper)."""
    from services.user_intent_executive import message_to_llm
    system_prompt = await prepare_terms_and_conditions_response(user_message)
    return await message_to_llm(system_prompt, " ", [])


async def prepare_terms_and_conditions_response(user_message: str) -> str:
    """Prepare a system prompt for terms, privacy, and policies."""
    document_context = get_document_context(user_message)
    return _build_rag_prompt(
        user_message, 
        document_context, 
        "about our store policies, terms, or privacy practices"
    )


async def prepare_store_info_response(user_message: str) -> str:
    """Prepare a system prompt for hours, locations, and facilities."""
    document_context = get_document_context(user_message)
    return _build_rag_prompt(
        user_message, 
        document_context, 
        "for store information such as hours, locations, parking, or delivery options"
    )


async def prepare_general_chat_response(user_message: str) -> str:
    """Prepare a system prompt for general chat and miscellaneous store inquiries."""
    document_context = get_document_context(user_message)
    return _build_rag_prompt(
        user_message, 
        document_context, 
        "a general question about UCC Supermarket or related services"
    )

