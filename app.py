import streamlit as st
import torch
from PIL import Image

# your modules
from inference import run_inference


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Radiology Report Generator",
    layout="wide"
)

st.title("Multimodal Radiology Report Generation")
st.write("Generate radiology reports from chest X-rays with factual validation.")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

image_file = st.sidebar.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)


# -----------------------------
# Main UI
# -----------------------------
if image_file is not None:

    # Display image
    image = Image.open(image_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input X-ray")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Generating Report...")

        with st.spinner("Running inference..."):

            # Save temp image
            temp_path = "temp_image.png"
            image.save(temp_path)

            result = run_inference(temp_path)

        st.success("Done!")

        # -----------------------------
        # Generated Report
        # -----------------------------
        st.subheader("Generated Report")
        st.write(result.generated_report)

        # -----------------------------
        # Retrieved Reports
        # -----------------------------
        st.subheader("Retrieved Reports")

        for i, (doc, score) in enumerate(
            zip(result.retrieved_docs, result.retrieval_scores), 1
        ):
            with st.expander(f"Report {i} (Similarity: {score:.3f})"):
                st.write(doc)

        # -----------------------------
        # Hallucination Analysis
        # -----------------------------
        st.subheader("Hallucination Analysis")

        h = result.hallucination

        st.write(f"Hallucination Rate: {h.hallucination_rate:.2%}")
        st.write(f"Unsupported Entities: {list(h.unsupported_entities)}")

        if h.is_flagged:
            st.error("Report may contain hallucinations")
        else:
            st.success("Report is factually consistent")


else:
    st.info("Please upload a chest X-ray image to begin.")