# 🔬 AI-Powered Document Text Restoration Pipeline

This project is an advanced, hybrid pipeline designed to restore and rebuild missing text from severely degraded or physically damaged documents (due to burns, tears, or aging). It combines **Computer Vision** with **Optical Character Recognition (OCR)** and **Large Language Models (LLMs)**.

## ⚙️ How It Works
1. **Physical Damage Mapping (CV):** Uses OpenCV grayscale thresholding and morphological operations to isolate burned or torn regions of the document.
2. **Text Detection (OCR):** Leverages PaddleOCR to detect text lines and calculate precise spatial bounding boxes.
3. **Contextual Text Reconstruction (LLM/NLP):** Cross-references text coordinates with the damage mask. If a line is damaged, the surrounding text is passed into **Mistral AI (via LangChain)** to intelligently predict and fill in the missing words based on semantic context.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Computer Vision:** OpenCV (cv2)
* **OCR Engine:** PaddleOCR & PaddlePaddle
* **LLM Framework:** LangChain & Mistral AI (`mistral-large-latest`)
* **User Interface:** Streamlit for web application deployment