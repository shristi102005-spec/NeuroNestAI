import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from modules.pdf_reader import extract_text
from modules.rag_engine import create_vector_store
import os

# Load environment variables
load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Title
st.title("NeuroNest AI")

# Sidebar
st.sidebar.title("NeuroNest AI")

# Clear chat button
if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# Initialize session state
if "messages" not in st.session_state:

    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# PDF Processing
if uploaded_file:

    # Create uploads folder
    os.makedirs("uploads", exist_ok=True)

    # Save PDF
    pdf_path = f"uploads/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully")

    # Extract text
    pdf_text = extract_text(pdf_path)

    # Debug
    st.write("PDF Text Length:", len(pdf_text))

    # Show preview
    st.subheader("PDF Preview")

    st.text_area(
        "Extracted Text",
        pdf_text[:3000],
        height=300
    )

    # Store PDF text
    st.session_state.pdf_text = pdf_text

    # Create vector database
    db = create_vector_store(pdf_text)

    st.session_state.vector_db = db

# Quiz Generator
if st.button("Generate Quiz from PDF"):

    if "pdf_text" in st.session_state:

        with st.spinner("Generating Quiz..."):

            quiz_prompt = f"""
            Generate 5 multiple choice questions
            from the following study material.

            Also provide correct answers.

            Study Material:
            {st.session_state.pdf_text[:4000]}
            """

            completion = client.chat.completions.create(

                model="deepseek/deepseek-chat",

                messages=[
                    {
                        "role": "user",
                        "content": quiz_prompt
                    }
                ]
            )

            quiz_response = completion.choices[0].message.content

            st.subheader("Generated Quiz")

            st.write(quiz_response)

    else:

        st.warning("Please upload a PDF first.")
        # Notes Generator
        if st.button("Generate Notes"):

            if "pdf_text" in st.session_state:

                with st.spinner("Generating Notes..."):

                    notes_prompt = f"""
                    Create clean study notes from the
                    following material.

                    Include:
                    - Important concepts
                    - Key points
                    - Short explanations
                    - Bullet format

                    Study Material:
                    {st.session_state.pdf_text[:5000]}
                    """

                    completion = client.chat.completions.create(

                        model="deepseek/deepseek-chat",

                        messages=[
                            {
                                "role": "user",
                                "content": notes_prompt
                            }
                        ]
                    )

                    notes_response = completion.choices[0].message.content

                    st.subheader("Generated Notes")

                    st.write(notes_response)

            else:

                st.warning("Please upload a PDF first.")

# Chat input
user_input = st.chat_input(
    "Ask something..."
)

# Chatbot
if user_input:

    # Show user message
    with st.chat_message("user"):

        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Use RAG if PDF exists
    if "vector_db" in st.session_state:

        results = st.session_state.vector_db.similarity_search(
            user_input,
            k=3
        )

        context = "\n".join(
            [r.page_content for r in results]
        )

        prompt = f"""
        Answer using the context below.

        Context:
        {context}

        Question:
        {user_input}
        """

    else:

        prompt = user_input

    # AI response
    completion = client.chat.completions.create(

        model="deepseek/deepseek-chat",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    ai_response = completion.choices[0].message.content

    # Show AI response
    with st.chat_message("assistant"):

        st.markdown(ai_response)

    # Save AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )