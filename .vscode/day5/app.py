import gradio as gr
from dotenv import load_dotenv
import os

# Assuming your answer_question function is in implementation/answer.py
from implementation.answer import answer_question

load_dotenv(override=True)

def format_context(context):
    """
    Formats the retrieved LangChain Documents into a clean HTML view.
    """
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        # Check if metadata exists to prevent crashes
        source = doc.metadata.get('source', 'Unknown Source')
        result += f"<b style='color: #ff7800;'>Source: {source}</b><br>"
        result += f"<p>{doc.page_content}</p><hr>"
    return result

def chat(history):
    # Gradio history is a list of dicts: [{"role": "user", "content": "..."}, ...]
    last_message = history[-1]["content"]
    
    # Standard LangChain/Groq history format
    prior_history = history[:-1]
    
    # Call your updated local-embedding + Groq function
    answer, context = answer_question(last_message, prior_history)
    
    # Append the assistant response to history
    history.append({"role": "assistant", "content": answer})
    
    return history, format_context(context)

def main():
    def put_message_in_chatbot(message, history):
        # Return an empty string for the textbox and append the user message to history
        return "", history + [{"role": "user", "content": message}]

    # Define theme here, but don't pass it to gr.Blocks
    theme = gr.themes.Soft(primary_hue="orange", font=["Inter", "sans-serif"])

    # 1. Removed 'theme' from gr.Blocks
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=2):
                # 2. Removed 'type="messages"' (it is now the default/only type)
                chatbot = gr.Chatbot(
                    label="💬 Conversation", 
                    height=600
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the Professional Tier features?",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.HTML(
                    label="📚 Retrieved Context",
                    value="<p style='color: gray;'>Retrieved context will appear here...</p>",
                )

        message.submit(
            put_message_in_chatbot, 
            inputs=[message, chatbot], 
            outputs=[message, chatbot]
        ).then(
            chat, 
            inputs=chatbot, 
            outputs=[chatbot, context_markdown]
        )

    # 3. Moved 'theme' here
    ui.launch(inbrowser=True, theme=theme)

if __name__ == "__main__":
    main()