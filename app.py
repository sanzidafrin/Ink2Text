
import streamlit as st
from PIL import Image
from src.handwriting_recognition import recognize_handwriting

st.set_page_config(
    page_title="Ink2Text",
    page_icon="📝",
    layout="wide"
)

st.title("Ink2Text")
st.subheader("English Handwritten Text Recognition")
st.write(
    "Upload an image of handwritten English text "
    "and convert it into editable digital text."
)

uploaded_file = st.file_uploader(
    "Upload a handwriting image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded handwriting",
        use_container_width=True
    )

    if st.button("Recognize Handwriting"):
        with st.spinner("Recognizing handwriting..."):
            try:
                recognized_text = recognize_handwriting(image)

                st.success("Recognition completed!")

                st.subheader("Recognized Text")

                edited_text = st.text_area(
                    "You can edit the recognized text:",
                    value=recognized_text,
                    height=180
                )

                st.download_button(
                    label="Download Text",
                    data=edited_text,
                    file_name="recognized_text.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Recognition failed: {e}")
