import api
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)

# llm
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=api.huggingfacehub_api_token,
)
chat_model = ChatHuggingFace(llm=llm)

# # single input prompt template
template = "suggest me a {lang} song"
prompt_template = ChatPromptTemplate.from_template(template=template)

prompt = prompt_template.invoke("tamil")

response = chat_model.invoke(prompt)
print(response.content)


# multiple input prompt template
template = "suggest me a {lang} song with {genre} genre"
prompt_template = ChatPromptTemplate.from_template(template=template)

prompt = prompt_template.invoke({"lang": "tamil", 
                                 "genre": "melody"})

response = chat_model.invoke(prompt)
print(response.content)


# prompt with system and human messages
messages = [
    ("system", "you are a {topic} provider"),
    ("human", "provide me a {topic} by {author}")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt = prompt_template.invoke({"topic": "song", 
                                 "author": "Anirudh Ravichandar"})
response = chat_model.invoke(prompt)
print(response.content)

# this will work
messages = [
    ("system", "you are a {topic} provider"),
    HumanMessage("provide me a song")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt = prompt_template.invoke({"topic": "song", 
                                 "author": "Anirudh Ravichandar"})
response = chat_model.invoke(prompt)
print(response.content)

# this will not work
messages = [
    ("system", "you are a {topic} provider"),
    HumanMessage("provide me a {topic}")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt = prompt_template.invoke({"topic": "song", 
                                 "author": "Anirudh Ravichandar"})
response = chat_model.invoke(prompt)
print(response.content)


