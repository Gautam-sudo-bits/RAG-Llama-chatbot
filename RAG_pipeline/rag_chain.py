import torch
from huggingface_hub import hf_hub_download
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# (Keep all other imports the same)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


DB_FAISS_PATH = 'vectorstore/db_faiss'
rag_chain = None

prompt_template_str = """
You are an expert research assistant. Your task is to provide detailed, accurate, and comprehensive answers based ONLY on the provided context. Do not use any of your prior knowledge.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1.  Synthesize the information from the context to answer the question thoroughly.
2.  If the context does not contain the answer, state clearly: "The provided documents do not contain enough information to answer this question."
3.  Quote relevant phrases or sentences from the context to support your answer where possible.
4.  Structure your answer in a clear, easy-to-read format. Use bullet points or numbered lists if it helps clarity.
5.  Do not make up any information. Your response must be grounded in the text provided.

ANSWER:
"""

def create_rag_chain():
    """
    Creates and returns a RetrievalQA chain that is hardware-aware.
    - If a GPU is available, it loads the powerful Llama 3 8B model.
    - If no GPU is available, it loads a CPU-optimized Llama 2 7B GGUF model.
    """
    global rag_chain
    if rag_chain:
        return rag_chain

    print("Loading the vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = None
    # --- 2. HARDWARE-AWARE MODEL LOADING ---
    if torch.cuda.is_available():
        # --- GPU PATH ---
        print("GPU detected. Loading Llama 3 8B model (approx. 16GB VRAM required)...")
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            top_k=10,
            temperature=0.7,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    else:
        print("No GPU detected. Loading CPU-friendly Llama 2 7B model (GGUF)...")
        
        model_file = "llama-2-7b-chat.Q4_K_M.gguf"
        model_path = hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename=model_file
        )

        # Load the GGUF model with LlamaCpp
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=0,      # Explicitly set to 0 to force CPU usage
            n_batch=512,         # Should be between 1 and n_ctx, consider memory constraints
            n_ctx=8192,          # Context window size
            max_tokens=512,      # Max tokens to generate
            f16_kv=True,         # Must be set to True
            verbose=False,       # Suppress verbose output
        )

    # --- CREATE THE RETRIEVALQA CHAIN (This is common) ---
    print("Creating the RAG chain with custom prompt...")

    PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "question"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return rag_chain