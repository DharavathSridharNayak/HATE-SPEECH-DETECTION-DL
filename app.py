import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Set page configuration
st.set_page_config(
    page_title="🚫 HATE SPEECH DETECTION",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #333333;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: pink;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff2b2b;
            color: pink;
        }
        .header {
            color: #ff4b4b;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subheader {
            color: #6c757d;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .result-box {
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: pink;
        }
        .safe {
            background-color: #d4edda;
            color: #155724;
        }
        .hate {
            background-color: #f8d7da;
            color: #721c24;
        }
        .neutral {
            background-color: #fff3cd;
            color: #856404;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load model (using a pre-trained model from Hugging Face)
@st.cache_resource
def load_model():
    model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

classifier = load_model()

# Function to classify text
def classify_text(text):
    if not text.strip():
        return None
    result = classifier(text)[0]
    return result['label'], result['score']

# App header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://img.icons8.com/color/96/000000/no-hate-speech.png", width=100)
with col2:
    st.markdown('<div class="header">🚫HATE SPEECH DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Advanced Hate Speech Detection System</div>', unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["Detector", "Information", "About"])

with tab1:
    st.markdown("### 🚨 Enter text to analyze for hate speech")
    user_input = st.text_area("", height=150, placeholder="Type or paste your text here...")
    
    if st.button("Analyze Text", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing text for hate speech..."):
                label, score = classify_text(user_input)
                time.sleep(1)  # Simulate processing time for better UX
                
                st.markdown("### Analysis Results")
                if label == "hate":
                    st.markdown(f"""
                        <div class="result-box hate">
                            <h3>🚫 Hate Speech Detected</h3>
                            <p>Confidence: {score:.2%}</p>
                            <p>This text contains harmful content that promotes hate or violence against individuals or groups based on attributes such as race, religion, ethnicity, sexual orientation, disability, or gender.</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif label == "nothate":
                    st.markdown(f"""
                        <div class="result-box safe">
                            <h3>✅ No Hate Speech Detected</h3>
                            <p>Confidence: {score:.2%}</p>
                            <p>This text appears to be free from hate speech content. However, always be mindful of the language you use and its potential impact on others.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-box neutral">
                            <h3>⚠️ Potentially Offensive Content</h3>
                            <p>Confidence: {score:.2%}</p>
                            <p>This text may contain language that some people could find offensive or inappropriate, though it doesn't meet the threshold for hate speech.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show explanation
                st.markdown("""
                    <div style="margin-top: 2rem; padding: 1rem; background-color: black; color: white; border-radius: 5px;">
                        <h4>Understanding the Results</h4>
                        <p>Our AI model analyzes text based on patterns learned from thousands of examples of hate speech and non-hate speech. While highly accurate, no system is perfect. Please use this as a guide, not an absolute determination.</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.markdown("""
        ## ℹ️ About Hate Speech Detection
        
        Hate speech detection is the process of identifying language that attacks or discriminates against individuals or groups based on attributes such as race, religion, ethnic origin, sexual orientation, disability, or gender.
        
        ### How It Works
        - Our system uses a state-of-the-art deep learning model (RoBERTa) trained on thousands of examples
        - The model analyzes the text for patterns commonly found in hate speech
        - Results are provided with a confidence score
        
        ### What Constitutes Hate Speech?
        Hate speech may include:
        - Direct threats or incitement to violence
        - Dehumanizing language
        - Harmful stereotypes
        - Slurs or derogatory terms targeting specific groups
        
        ### Limitations
        - Context matters - sarcasm or reclaimed words may be misinterpreted
        - New forms of hate speech may emerge that the model hasn't seen
        - Cultural differences in what constitutes hate speech
    """)

with tab3:
    st.markdown("""
        ## 🚀 🚫 HATE SPEECH DETECTION
        
        Hate Speech Detection is an advanced hate speech detection system designed to help identify potentially harmful content in text.
        
        ### Features
        - Real-time text analysis
        - Confidence scoring
        - Educational resources about hate speech
        
        ### Technology Stack
        - Built with Python and Streamlit
        - Powered by Hugging Face Transformers
        - Uses RoBERTa model fine-tuned for hate speech detection
        
        ### Disclaimer
        This tool is intended for educational and research purposes only. The developers are not responsible for decisions made based on this tool's output.
        
        ### Contact
        For questions or feedback, please contact: support@hatescope.ai
    """)
    
    st.markdown('<div class="footer">© 2025 Hate Speech Detection | Made with ❤️ for a better internet</div>', unsafe_allow_html=True)
