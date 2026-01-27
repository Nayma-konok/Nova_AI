#from dotenv import load_dotenv
import os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser   
from langchain_google_genai import ChatGoogleGenerativeAI

#load_dotenv()

gemini_key = os.environ.get("Gemini_Api_Key")

if not gemini_key:
    raise ValueError("Gemini_Api_Key not found. Check Hugging Face Secrets.")

system_prompt="""You are Nova, a friendly, witty, and intelligent AI assistant.  
You can answer any question the user asks — from serious topics to silly curiosities — 
with clarity, confidence, and a touch of humor.Keep your tone light, engaging, and positive.  
If the topic allows, sprinkle in a clever joke or playful remark, but never be rude or sarcastic.  
Your goal is to make the user smile while giving genuinely helpful and accurate answers.
ANswer in 2-4 lines.
"""

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user","{input}")
])

chain= prompt | llm | StrOutputParser()

def Chat(user_input,history):
    langchain_history=[]
    for item in history:
        if item["role"]=="user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"]=="assistant":
            langchain_history.append(AIMessage(content=item["content"]))
    response=chain.invoke({"input":user_input,"history":langchain_history})

    return "", history+[{"role":"user",'content':user_input},
                        {"role":"assistant",'content':response}]


def clear_chat():
    return "",[]

image_path = "Nova_Image.png"
if not os.path.exists(image_path):
    print(f"Warning: {image_path} not found! Chatbot will load without avatar.")

page = gr.Blocks(title="SuperNova")

with page:
    gr.Markdown(
        """
        # Chat With Nova
        Welcome to your personal conversation with Nova
        """
    )
    chatbot = gr.Chatbot(
        avatar_images=[None, image_path if os.path.exists(image_path) else None],
        show_label=False,
        type="messages"  # fixes deprecated warning
    )
    msg=gr.Textbox(show_label=False,
                   placeholder="Ask Nova anything...")

    msg.submit(Chat, [msg, chatbot], [msg, chatbot])

    Clear=gr.Button("Clear Chat",variant="secondary")
    Clear.click(clear_chat,outputs=[msg, chatbot])

port = int(os.environ.get("PORT", 7860))
url = f"http://127.0.0.1:{port}"
print(f"Nova AI is running locally at: {url}")

page.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=True
)