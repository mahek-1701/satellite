import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful colorful styling
st.markdown("""
<style>
    /* Global styling */
    .main .block-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        border: 2px dashed rgba(255,255,255,0.5);
        text-align: center;
        transition: all 0.3s ease;
        color: white;
    }
    
    .upload-section:hover {
        border-color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .upload-section h3 {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Results section */
    .results-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.15);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: white;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
        opacity: 0.9;
        color: white;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        color: white;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .feature-card h4 {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .feature-card p {
        color: rgba(255,255,255,0.9);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .sidebar-info h3 {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .sidebar-info p {
        color: rgba(255,255,255,0.9);
    }
    
    /* Override Streamlit default styles */
    .stMarkdown {
        color: white;
    }
    
    /* Image info section */
    .image-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Spinner styling */
    .stSpinner {
        text-align: center;
    }
    
    /* Footer styling */
    .footer-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Model download link (Google Drive)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🤖 Downloading AI model... (this happens only once)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

# Load model
model = download_and_load_model()

# Class labels with emojis
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
class_emojis = {'Cloudy': '☁️', 'Desert': '🏜️', 'Green_Area': '🌿', 'Water': '💧'}
class_colors = {'Cloudy': '#87CEEB', 'Desert': '#DEB887', 'Green_Area': '#90EE90', 'Water': '#87CEFA'}

# Main header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>Advanced AI-powered classification of satellite imagery using deep learning</p>
</div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Upload section
    st.markdown("""
    <div class="upload-section">
        <h3>📤 Upload Your Satellite Image</h3>
        <p>Drag and drop or click to upload a satellite image (JPG, JPEG, PNG)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear satellite image for best results"
    )

# Sidebar with information
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h3>🔍 About This Classifier</h3>
        <p>This AI model can identify four types of terrain from satellite images:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Classification Types")
    for class_name in class_names:
        st.markdown(f"**{class_emojis[class_name]} {class_name.replace('_', ' ')}**")
    
    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1. **Upload** a satellite image
    2. **AI Analysis** processes the image
    3. **Classification** identifies terrain type
    4. **Results** show prediction with confidence
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Use clear, high-quality images
    - Ensure good lighting conditions
    - Images should be primarily of one terrain type
    - Avoid heavily processed or filtered images
    """)

# Main content area
if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    image = image.resize((256, 256))
    
    # Display image and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, caption=f"Original size: {original_size[0]}x{original_size[1]} pixels", use_container_width=True)
        
        # Image info
        st.markdown(f"""
        <div class="image-info">
            <h4>📊 Image Information</h4>
            <p><strong>File name:</strong> {uploaded_file.name}</p>
            <p><strong>File size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
            <p><strong>Dimensions:</strong> {original_size[0]} x {original_size[1]} pixels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Preprocess image
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        with st.spinner("🔮 Analyzing image with AI..."):
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
        
        # Display results
        st.markdown(f"""
        <div class="results-section">
            <div class="metric-card">
                <div class="prediction-text">
                    {class_emojis[predicted_class]} {predicted_class.replace('_', ' ')}
                </div>
                <div class="confidence-text">
                    Confidence: {confidence * 100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence level indicator
        if confidence > 0.8:
            st.success("🎯 High confidence prediction!")
        elif confidence > 0.6:
            st.warning("⚠️ Moderate confidence prediction")
        else:
            st.error("❓ Low confidence - image may be unclear")
    
    # Detailed prediction results
    st.markdown("### 📈 Detailed Prediction Analysis")
    
    # Create prediction chart
    fig = go.Figure(data=[
        go.Bar(
            x=[name.replace('_', ' ') for name in class_names],
            y=prediction * 100,
            marker_color=[class_colors[name] for name in class_names],
            text=[f"{p*100:.1f}%" for p in prediction],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence for Each Class",
        xaxis_title="Terrain Type",
        yaxis_title="Confidence (%)",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction breakdown
    st.markdown("### 🔍 Confidence Breakdown")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: white; text-align: center; margin-bottom: 1rem;">Individual Class Predictions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    for i, (class_name, prob) in enumerate(zip(class_names, prediction)):
        is_predicted = i == np.argmax(prediction)
        bg_color = "linear-gradient(135deg, #00b894 0%, #00a085 100%)" if is_predicted else "linear-gradient(135deg, #636e72 0%, #2d3436 100%)"
        
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <h5 style="color: white; margin-bottom: 0.5rem;">{class_emojis[class_name]} {class_name.replace('_', ' ')}</h5>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob), text=f"{prob*100:.1f}%")

else:
    # Feature showcase when no image is uploaded
    st.markdown("### ✨ Key Features")
    
    feature_cols = st.columns(4)
    
    features = [
        ("🤖", "AI-Powered", "Deep learning model trained on satellite imagery"),
        ("⚡", "Fast Processing", "Get results in seconds"),
        ("🎯", "High Accuracy", "Reliable terrain classification"),
        ("🌍", "Multiple Terrains", "Detects clouds, desert, vegetation, and water")
    ]
    
    for i, (icon, title, description) in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sample images section
    st.markdown("### 📸 Sample Images")
    st.markdown("Try uploading satellite images containing these terrain types:")
    
    sample_cols = st.columns(4)
    sample_descriptions = [
        ("☁️ **Cloudy Areas**", "Cloud formations over landscapes", "linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)"),
        ("🏜️ **Desert Regions**", "Arid landscapes and sand dunes", "linear-gradient(135deg, #fdcb6e 0%, #e17055 100%)"), 
        ("🌿 **Green Areas**", "Forests, fields, and vegetation", "linear-gradient(135deg, #00b894 0%, #00cec9 100%)"),
        ("💧 **Water Bodies**", "Oceans, lakes, and rivers", "linear-gradient(135deg, #81ecec 0%, #74b9ff 100%)")
    ]
    
    for i, (title, desc, gradient) in enumerate(sample_descriptions):
        with sample_cols[i]:
            st.markdown(f"""
            <div style="background: {gradient}; padding: 1.5rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 1rem;">
                <h5 style="color: white; margin-bottom: 0.5rem;">{title}</h5>
                <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-gradient">
    <h4>🛰️ Satellite Image Classifier</h4>
    <p>Powered by TensorFlow and Streamlit | Built with ❤️ for Earth observation</p>
</div>
""", unsafe_allow_html=True)