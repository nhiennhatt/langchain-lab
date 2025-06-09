import os

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()


def load_llm():
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0.5
    )
    return llm


def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    return prompt


def create_chain(prompt, llm, retriever):
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain


def create_retriever():
    embeddings = AzureOpenAIEmbeddings(azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"])
    vector_stores = FAISS.load_local(folder_path="azure_openai/vector_stores/db_faiss", embeddings=embeddings,
                                     allow_dangerous_deserialization=True)
    return vector_stores.as_retriever()


template = """You're a learning assistant. You're expert in Software Engineering. You understand English and Vietnamese.  
You answer the question. You're answer base on this information: {context}.  
{question}
"""

prompt = create_prompt(template)
llm_ = load_llm()
retriever = create_retriever()
chain = create_chain(prompt, llm_, retriever)

question = "Kiểm thử hộp trắng là gì?"
response = chain.invoke(question)
print(response)
