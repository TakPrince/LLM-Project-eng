import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(override=True)

# Use a Groq model (e.g., Llama 3.3 or Mixtral)
MODEL = "llama-3.3-70b-versatile"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# 🔁 REPLACED: Using local HuggingFace embeddings instead of OpenAI
# 'all-MiniLM-L6-v2' is a fast, free local model.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Ensure the vectorstore uses the new embedding function
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# 🔁 REPLACED: Using Groq for the LLM
llm = ChatGroq(
    temperature=0, 
    model_name=MODEL,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def fetch_context(question: str) -> list[Document]:
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    Correctly handles Gradio 6.0's list-based content format.
    """
    user_texts = []
    
    for m in history:
        if m.get("role") == "user":
            content = m.get("content", "")
            # Gradio 6.0: content is a list of dicts like [{"type": "text", "text": "..."}]
            if isinstance(content, list):
                text_parts = [item["text"] for item in content if item.get("type") == "text"]
                user_texts.append(" ".join(text_parts))
            else:
                # Fallback for simple string if Gradio version varies
                user_texts.append(str(content))
    
    prior = "\n".join(user_texts)
    return prior + "\n" + question

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    # 1. Combine question and history for retrieval
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    system_prompt = SYSTEM_PROMPT.format(context=context)
    
    # 2. Re-map history for LangChain to avoid "list vs str" issues
    langchain_history = []
    for msg in history:
        content = msg["content"]
        # Extract the actual text from the Gradio 6 list structure
        if isinstance(content, list):
            text = " ".join([item["text"] for item in content if item.get("type") == "text"])
        else:
            text = content
            
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=text))
        elif msg["role"] == "assistant":
            langchain_history.append(AIMessage(content=text))

    messages = [SystemMessage(content=system_prompt)] + langchain_history + [HumanMessage(content=question)]
    
    # 3. Get LLM response
    response = llm.invoke(messages)
    return response.content, docs