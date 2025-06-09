from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def load_llm():
    llm = CTransformers(
        model="models/vinallama-7b-chat-GGUF/vinallama-7b-chat_q5_0.gguf",
        model_type="llama"
    )
    return llm


def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt


def create_chain(prompt, llm):
    chain = prompt | llm
    return chain


template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm_ = load_llm()
chain = create_chain(prompt, llm_)

question = "Một cộng một bằng mấy?"
response = chain.invoke({'question': question})
print(response)
