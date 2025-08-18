import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG_pipeline.rag_chain import create_rag_chain

def load_css():
    """Injects custom CSS for a professional look and feel."""
    st.markdown("""
        <style>
            /* Main app background with the 4-color gradient */
            .stApp {
                background: linear-gradient(-45deg, #ff66c4, #ff8c59, #9370db, #23d5ab);
                background-attachment: fixed;
                background-size: 400% 400%;
                animation: pan-gradient 15s ease infinite;
            }

            @keyframes pan-gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Chat bubble styling */
            /* Using semi-transparent dark bubbles for a modern look on the new gradient */
            .st-emotion-cache-1c7y2kd, .st-emotion-cache-4k6c4v {
                background-color: rgba(0, 0, 0, 0.2);
                color: #FFFFFF; /* Makes text in both bubbles white */
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }

            /* Title styling (White to be visible on the gradient) */
            .st-emotion-cache-10trblm, .st-emotion-cache-1f1f6c, h1, h3 {
                color: #FFFFFF !important;
                text-shadow: 0 0 5px rgba(0,0,0,0.5);
            }

            /* Sidebar styling */
            .st-emotion-cache-16txtl3 {
                 background-color: rgba(30, 41, 59, 0.7);
                 backdrop-filter: blur(10px);
                 border-right: 1px solid rgba(255, 255, 255, 0.1);
            }

        </style>
    """, unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Assistant: Ask anything about RLHF",
    layout="wide"
)

load_css()

with st.sidebar:
    st.header("About")
    st.info(
        "This application is an AI-powered assistant that allows you to chat "
        "with your documents. It uses a Retrieval-Augmented Generation (RAG) "
        "pipeline to provide contextually accurate answers related to RLHF."
    )
    st.markdown("---")
    st.header("How It Works")
    st.markdown(
        """
        1.  **Upload Documents**: The system processes PDFs from a local `data` folder.
        2.  **Ask a Question**: Type your query into the chat input.
        3.  **Get Answers**: The AI retrieves relevant information from the documents and generates a comprehensive answer.
        """
    )


st.title("AI Assistant")
st.subheader("Your intelligent partner for RLHF insights")

@st.cache_resource
def load_chain():
    with st.spinner("Initializing the AI engine... Please wait."):
        return create_rag_chain()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about RLHF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke({'query': prompt})
            answer = result['result']
            source_docs = result['source_documents']
            
            st.markdown(answer)

            with st.expander("View Sources"):
                for doc in source_docs:
                    source_path = doc.metadata.get('source', 'N/A').replace('\\', '/')
                    file_name = os.path.basename(source_path)
                    page_number = doc.metadata.get('page', 'N/A')
                    st.write(f"**File:** {file_name}  |  **Page:** {page_number}")
                    # Optionally display a snippet of the source content
                    # st.caption(f"Content: \"{doc.page_content[:150]}...\"")
            
            # Add the complete response (answer + sources) to history
            full_response = {"role": "assistant", "content": answer}
            st.session_state.messages.append(full_response)