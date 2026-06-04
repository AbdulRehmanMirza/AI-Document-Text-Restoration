import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from paddleocr import PaddleOCR
from langchain_mistralai import ChatMistralAI

# Streamlit Page Design Setup
st.set_page_config(layout="wide")
st.title("🔬 Text Restoration & Document Repair App")
st.write("Welcome Fayz! Yeh app Computer Vision aur Mistral AI use karke damaged documents restore karti hai.")

# 1. Aapki Mistral AI Key (Hardcoded as you requested)
import os
api_key = os.environ.get("MISTRAL_API_KEY", "")
llm = ChatMistralAI(api_key=api_key, model="mistral-large-latest")

# Helper functions aapke notebook se
def is_damaged(bbox, mask):
    """Check if text bounding box overlaps with damage mask"""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    roi = mask[y1:y2, x1:x2]
    return np.mean(roi) > 10

def reconstruct_text(context, llm):
    """AI reconstruction of damaged text using context"""
    prompt = f"""
    The following document text has damaged or missing words.

    Context:
    {context}

    Predict the missing text accurately and naturally.

    Give me short, accurate answers.
    Only return me the complete paragraph exactly written in image and read by OCR. Fine tune how you read it and only give me the paragraph, fixing the missing spaces/words, also show it with asterisks the specific missing words. After you return the paragraph, show a short reason why you filled it with that word (depending upon context and author behavior).
    """
    return llm.invoke(prompt).content

# Main Pipeline function (Adjusted for local file uploads)
def process_document_local(image_bytes):
    # Image bytes ko OpenCV format mein decode karna
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    original = image.copy()

    # Detect damage (CV2 thresholding)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black/burn detection
    _, black_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((3,3)))

    # Tear/paper loss detection
    _, tear_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    tear_mask = cv2.morphologyEx(tear_mask, cv2.MORPH_CLOSE, np.ones((5,5)))

    # Combine masks
    damage_mask = cv2.bitwise_or(black_mask, tear_mask)

    # OCR extraction using PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    ocr_result = ocr.ocr(image, cls=True)

    ocr_data = []
    if ocr_result and ocr_result[0]:
        for box, (text, conf) in ocr_result[0]:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            ocr_data.append({
                "text": text,
                "confidence": conf,
                "bbox": [min(xs), min(ys), max(xs), max(ys)]
            })

    damaged_lines = []
    clean_lines = []

    for item in ocr_data:
        if is_damaged(item["bbox"], damage_mask):
            damaged_lines.append(item)
        else:
            clean_lines.append(item)

    final_output = []
    texts = [x["text"] for x in ocr_data]

    for i, item in enumerate(ocr_data):
        if item in damaged_lines:
            context = " ".join(texts[max(0, i-2):i+3])
            fixed = reconstruct_text(context, llm)
        else:
            fixed = item["text"]

        final_output.append({
            "original": item["text"],
            "reconstructed": fixed,
            "confidence": item["confidence"]
        })

    # Create 3-panel visualization chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Document")
    axes[0].axis("off")

    axes[1].imshow(damage_mask, cmap="gray")
    axes[1].set_title("Detected Damage Mask")
    axes[1].axis("off")

    damage_view = original.copy()
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 300:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(damage_view, (x,y), (x+w,y+h), (0,0,255), 2)

    axes[2].imshow(cv2.cvtColor(damage_view, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detected Damaged Areas")
    axes[2].axis("off")
    plt.tight_layout()

    return final_output, fig, len(ocr_data), len(damaged_lines), len(clean_lines)

# Front-end components
uploaded_file = st.file_uploader("Apni damaged project document image upload karein:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    with st.spinner("AI Pipeline running... Drawing Computer Vision Masks..."):
        try:
            final_output, fig, total_lines, damaged_cnt, clean_cnt = process_document_local(file_bytes)
            
            # Display plots on web UI
            st.pyplot(fig)
            
            # Show summary
            st.subheader("📊 Execution Statistics")
            st.write(f"**Total Extracted Text Lines:** {total_lines}")
            st.write(f"**Damaged Lines Intersected:** {damaged_cnt}")
            st.write(f"**Clean Lines Retained:** {clean_cnt}")
            
            # Print row iterations
            st.subheader("📝 Restored Output")
            for i, row in enumerate(final_output):
                st.markdown(f"### 🔹 Line {i+1}")
                st.write(f"**Original:** {row['original']}")
                st.write(f"**Reconstructed:** {row['reconstructed']}")
                st.write(f"**Confidence:** {row['confidence']:.2f}%")
                st.write("---")
        except Exception as e:
            st.error(f"Execution Error: {e}")