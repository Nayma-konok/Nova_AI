from dotenv import load_dotenv
import os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser   
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key=os.getenv("Gemini_Api_Key")

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


page=gr.Blocks(
    title="SuperNova",
    theme=gr.themes.Soft(),
)

with page:
    gr.Markdown(
        """
        # Chat With Nova
        Welcome to your personal conversation with Nova
        """
)
    chatbot=gr.Chatbot(type="messages",
                       avatar_images=[None,"supernova_1993J.jpg"],
                       show_label=False)

    msg=gr.Textbox(show_label=False,
                   placeholder="Ask Nova anything...")

    msg.submit(Chat, [msg, chatbot], [msg, chatbot])

    Clear=gr.Button("Clear Chat",variant="secondary")
    Clear.click(clear_chat,outputs=[msg, chatbot])

page.launch(share=True)