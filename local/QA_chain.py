from langchain.prompts import PromptTemplate
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

embedding_model_path="models/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2.Q8_0.gguf"
model_path="models/vinallama-7b-chat-GGUF/vinallama-7b-chat_q5_0.gguf"

def load_llm():
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={
            'max_new_tokens': 512,
            'temperature': 0.7,
            'context_length': 4096  # Set the context length
        }
    )
    return llm


def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    return prompt


def create_chain(prompt, llm, retriever):
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    return chain


def create_retriever():
    embeddings = LlamaCppEmbeddings(model_path=embedding_model_path)
    vector_stores = FAISS.load_local(folder_path="local/vector_stores/db_faiss", embeddings=embeddings,
                                     allow_dangerous_deserialization=True)
    return vector_stores.as_retriever()


template = """<|im_start|>system
Bạn là một trợ lí AI cho việc học kiểm thử (Testing). Dựa vào thông tin sau đây để trả lời câu hỏi, nếu không biết đừng cố tạo ra thông tin:
{context}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm_ = load_llm()
retriever = create_retriever()
chain = create_chain(prompt, llm_, retriever)

question = "Kiểm thử hộp trắng là gì?"
response = chain.invoke({'question': question})
print(response)
