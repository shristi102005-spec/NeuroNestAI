if uploaded_file:

    pdf_path = f"uploads/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.success("PDF Uploaded Successfully")

    pdf_text = extract_text(pdf_path)
    st.write(pdf_text[:1000])