import api
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.base import RunnableLambda, RunnableSequence, RunnableParallel

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


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a {topic} provider"),
        ("human", "provide me a {topic} name alone")
    ]
)

def first_line_prompt(user_topic):
    prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "you are a {topic} provider"),
                    ("human", "provide me a '{user_topic}' song first line")
                ]
            )
    return prompt.format(user_topic=user_topic)

def last_line_prompt(user_topic):
    prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "you are a {topic} provider"),
                    ("human", "provide me a '{user_topic}' song last line")
                ]
            )
    return prompt.format(user_topic=user_topic)



first_line_chain = RunnableLambda(lambda x: first_line_prompt(x)) | chat_model | StrOutputParser()
last_line_chain = RunnableLambda(lambda x: last_line_prompt(x)) | chat_model | StrOutputParser()

def combine(firstline, lastline):
    return firstline + "\n\n" + lastline


# Langchain expression language (LCEL)
# output of one is passed as input to next
chain = (
    prompt_template 
    | chat_model 
    | StrOutputParser() 
    | RunnableParallel(branches={"firstline": first_line_chain, "lastline": last_line_chain})
    | RunnableLambda(lambda x: combine(x["branches"]["firstline"], x["branches"]["lastline"]))
)

response = chain.invoke({"topic": "song"})
print(response)