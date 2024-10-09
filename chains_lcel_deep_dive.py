import api
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.base import RunnableLambda, RunnableSequence

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
        ("human", "provide me a {topic} by {author}")
    ]
)

# under the hood of LCEL
prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
response = RunnableLambda(lambda x: chat_model.invoke(x.to_messages()))
response_parser = RunnableLambda(lambda x: x.content)
chain = RunnableSequence(first=prompt, middle=[response], last=response_parser)


response = chain.invoke({"topic": "song", 
                         "author": "A R Rahman"})
print(response)