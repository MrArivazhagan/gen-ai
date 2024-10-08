import api
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

# llm
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=api.huggingfacehub_api_token,
)
chat_model = ChatHuggingFace(llm=llm)

# history
chat_convo = [
    SystemMessage(content="You're a helpful assistant"),
]

# continuing the convo using loop same as chatgpt-chat
while True:
    user_prompt = input("Enter your prompt: ")
    if user_prompt.strip().lower() == "exit":
        break

    chat_convo.append(HumanMessage(user_prompt.strip()))
    response = chat_model.invoke(chat_convo)
    print(response.content)
    chat_convo.append(AIMessage(response.content))

print("-------- chat history ------------")
print(chat_convo)