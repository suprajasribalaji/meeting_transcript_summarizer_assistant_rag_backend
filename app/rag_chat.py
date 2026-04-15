from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.schema import Document, SystemMessage, HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

import chainlit as cl
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chainlit chat start
@cl.on_chat_start
async def chat_start():
    cl.user_session.set("file_uploaded", False)
    cl.user_session.set("vector_store", None)

    await cl.Message(
        content="👋 Welcome! Upload your meeting transcription PDF to begin."
    ).send()

# Chat message handler
@cl.on_message
async def handle_message(message: cl.Message):

    # Handle PDF in vector_store
    if message.elements:
        documents = []

        for element in message.elements:
            if element.mime == "application/pdf":
                reader = PdfReader(element.path)

                for page_number, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""

                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": element.name,
                                "page": page_number + 1
                            }
                        )
                    )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
        )

        split_docs = splitter.split_documents(documents)

        vector_store = FAISS.from_documents(split_docs, embeddings)

        cl.user_session.set("vector_store", vector_store)
        cl.user_session.set("file_uploaded", True)

        await cl.Message(
            content="✅ PDF uploaded indexed successfully! Ask anything related to your file"
        ).send()

        return

    # Normal Flow
    file_uploaded = cl.user_session.get("file_uploaded")
    vector_store = cl.user_session.get("vector_store")

    user_input = (message.content or "").strip()

    # If no PDF uploaded
    if not file_uploaded:
        await cl.Message(
            content="📄 Please upload a meeting transcription PDF first."
        ).send()
        return

    # If user sends PDF with no input
    if not user_input:
        user_input = "Provide a concise summary of the meeting in 5 bullet points."

    # Retrieval from the vector_store
    docs = vector_store.similarity_search(user_input, k=5)

    retrieved_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt for the assistant
    system_prompt = f"""
        You are an AI Meeting Transcription Assistant using RAG.

        Rules:
        - Use ONLY the retrieved context below to answer.
        - If format not specified, give 5 concise bullet points.
        - If format specified, follow exactly.
        - If answer not found, say "I don't know."
        - Do not hallucinate.
        - Keep response professional and concise.

        Retrieved Context:
        {retrieved_context}
    """

    async with cl.Step(name="Retrieving & Analyzing...", type="llm") as step:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])
        step.output = response.content

    await cl.Message(content=response.content).send()

@cl.on_chat_end
async def chat_end():
    print("The user disconnected!")