import api
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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

ai_msg = chat_model.invoke("meaning of love")
print(ai_msg.content)
