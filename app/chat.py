from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema import SystemMessage, HumanMessage
from pypdf import PdfReader
import chainlit as cl
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)


@cl.on_chat_start
async def chat_start():
    cl.user_session.set("file_uploaded", False)
    cl.user_session.set("transcript", None)

    await cl.Message(
        content="👋 Welcome! Upload your meeting transcription PDF to begin."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):

    # -------------------------------
    # HANDLE FILE UPLOAD FIRST
    # -------------------------------
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":
                reader = PdfReader(element.path)
                text = ""

                for page in reader.pages:
                    text += page.extract_text() or ""

                cl.user_session.set("transcript", text)
                cl.user_session.set("file_uploaded", True)

                await cl.Message(
                    content="✅ PDF uploaded successfully! Generating summary..."
                ).send()

    # -------------------------------
    # NORMAL MESSAGE FLOW
    # -------------------------------

    file_uploaded = cl.user_session.get("file_uploaded")
    transcript = cl.user_session.get("transcript")

    user_input = (message.content or "").strip()

    # If no PDF uploaded
    if not file_uploaded:

        system_prompt = """
            You are an AI Meeting Transcription Assistant.

            Rules:
            - Ask the user to upload a meeting transcription PDF.
            - If asked about capabilities, explain:
            • You summarize meeting transcripts
            • Extract action items
            • Highlight key decisions
            • Create structured meeting notes
            • Provide customized summaries
            - If unrelated question, say you specialize only in transcript analysis.
            - Be professional and concise.
        """

        async with cl.Step(name="Thinking...", type="llm") as step:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input or "Hello")
            ])
            step.output = response.content

        await cl.Message(content=response.content).send()
        return

    # -------------------------------
    # PDF ALREADY UPLOADED
    # -------------------------------

    system_prompt = f"""
        You are an AI Meeting Transcription Summarizing Assistant.

        Rules:
        - If the user does NOT specify any format or length,
        provide EXACTLY 5 concise summary lines,
        then ask what they want to know.

        - If the user specifies the number of lines, bullet points, 
        summary format or anything, follow the user's instructions exactly then ask
        what they want to know futher or next step.

        - If the user asks a question,
        answer concisely based STRICTLY ONLY on the transcript.

        - If you dont know the answer, say you dont know instead of making assumptions.

        - Do NOT add unnecessary information.

        - Keep responses professional.

        - Avoid hallucinating or making assumptions, jargons, or filler words.

        Transcript:
        {transcript}
    """

    async with cl.Step(name="Analyzing Transcript...", type="llm") as step:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input or "Summarize the meeting.")
        ])
        step.output = response.content

    await cl.Message(content=response.content).send()

@cl.on_chat_end
async def chat_end():
    print("The user disconnected!")