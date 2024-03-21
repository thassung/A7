import torch.nn as nn
import torch
from langchain import PromptTemplate, HuggingFacePipeline
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
import os

# import torch.nn as nn
# import torch
# from langchain import PromptTemplate, HuggingFacePipeline
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
# from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig
# from langchain.chains import LLMChain
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import ConversationalRetrievalChain
# import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = '..'

prompt_template = """
    I'm your friendly NLP chatbot named ChakyBot, here to assist Chaky and Gun with any questions they have about Natural Language Processing (NLP).
    If you're curious about how probability works in the context of NLP, feel free to ask any questions you may have.
    Whether it's about probabilistic models, language models, or any other related topic,
    I'm here to help break down complex concepts into easy-to-understand explanations.
    Just let me know what you're wondering about, and I'll do my best to guide you through it!

    Context: {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

nlp_docs = f'{file_path}/data/pdf/Student-Handbook_August-2021.pdf'

loader = PyMuPDFLoader(nlp_docs)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100
)

doc = text_splitter.split_documents(documents)

model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : device}
)

vector_path = f'{file_path}/vector-store'
if not os.path.exists(vector_path):
    os.makedirs(vector_path)
    print('create path done')

vectordb = FAISS.from_documents(
    documents = doc,
    embedding = embedding_model
)

db_file_name = 'nlp_stanford'

vectordb.save_local(
    folder_path = os.path.join(vector_path, db_file_name),
    index_name = 'nlp' #default index
)

vectordb = FAISS.load_local(
    folder_path = os.path.join(vector_path, db_file_name),
    embeddings = embedding_model,
    index_name = 'nlp' #default index
)

retriever = vectordb.as_retriever()

history = ChatMessageHistory()

model_id = f'{file_path}/model/fastchat-t5-3b-v1.0/'

tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

bitsandbyte_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config = bitsandbyte_config, #caution Nvidia
    device_map = 'auto',
    load_in_8bit = True
)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 256,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

memory = ConversationBufferWindowMemory(
    k=3,
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)

def query_to_response(query):
    answer = chain({"question":query})
    return answer['answer']

