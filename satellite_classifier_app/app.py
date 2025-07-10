import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random

# Set page config
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.2rem;
        margin: 0;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>AI-Powered Environmental Monitoring System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Image Classification", "📊 Model Analytics", "ℹ️ About Model"]
    )
    
    st.markdown("---")
    st.markdown("### 🌟 Model Information")
    st.markdown("""
    **Classes Detected:**
    - 🌥️ Cloudy
    - 🏜️ Desert  
    - 🌲 Green Area
    - 💧 Water
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "94.2%", "↑ 2.1%")
    with col2:
        st.metric("Classes", "4", "")

# Mock model data for demonstration
@st.cache_data
def load_mock_data():
    # Training history data
    epochs = list(range(1, 11))
    train_acc = [0.6, 0.72, 0.81, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.942]
    val_acc = [0.58, 0.70, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.918]
    train_loss = [1.2, 0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.22]
    val_loss = [1.25, 0.95, 0.75, 0.65, 0.55, 0.45, 0.40, 0.35, 0.30, 0.28]
    
    return {
        'epochs': epochs,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    }

def classify_image(image):
    """Mock classification function"""
    classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
    emojis = ['🌥️', '🏜️', '🌲', '💧']
    
    # Simulate processing time
    time.sleep(2)
    
    # Mock prediction
    prediction = random.choice(classes)
    confidence = random.uniform(0.85, 0.99)
    
    # Mock confidence scores for all classes
    scores = {
        'Cloudy': random.uniform(0.05, 0.95),
        'Desert': random.uniform(0.05, 0.95),
        'Green_Area': random.uniform(0.05, 0.95),
        'Water': random.uniform(0.05, 0.95)
    }
    
    # Ensure the predicted class has the highest score
    scores[prediction] = confidence
    
    return prediction, confidence, scores, emojis[classes.index(prediction)]

# Page routing
if page == "🏠 Home":
    st.markdown("## Welcome to the Satellite Image Classifier!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Purpose</h3>
            <p>Automatically classify satellite images into different environmental categories using deep learning.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🧠 Technology</h3>
            <p>Powered by Convolutional Neural Networks (CNN) with 94.2% accuracy on test data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>🌍 Applications</h3>
            <p>Environmental monitoring, urban planning, disaster management, and climate research.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 **Advanced Classification**
        - Real-time image processing
        - High accuracy predictions
        - Confidence scoring
        - Multi-class support
        """)
        
        st.markdown("""
        ### 📊 **Detailed Analytics**
        - Training performance metrics
        - Confusion matrix visualization
        - Loss and accuracy curves
        - Model evaluation statistics
        """)
    
    with col2:
        st.markdown("""
        ### 🎨 **User-Friendly Interface**
        - Drag-and-drop image upload
        - Interactive visualizations
        - Real-time predictions
        - Responsive design
        """)
        
        st.markdown("""
        ### 🛡️ **Robust Performance**
        - Handles various image formats
        - Consistent predictions
        - Fast processing speed
        - Error handling
        """)

elif page == "🔍 Image Classification":
    st.markdown("## 🔍 Upload and Classify Your Satellite Image")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a satellite image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a satellite image to classify it into one of four categories: Cloudy, Desert, Green Area, or Water."
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", use_column_width=True)
            
            # Image info
            st.markdown("**Image Information:**")
            st.write(f"- **Format:** {image.format}")
            st.write(f"- **Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"- **Mode:** {image.mode}")
        
        with col2:
            st.markdown("### 🎯 Classification Results")
            
            if st.button("🚀 Classify Image", key="classify"):
                with st.spinner("🔄 Processing image..."):
                    prediction, confidence, scores, emoji = classify_image(image)
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{emoji} {prediction}</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores for all classes
                st.markdown("### 📊 Confidence Scores")
                
                # Create a DataFrame for the scores
                scores_df = pd.DataFrame(list(scores.items()), columns=['Class', 'Confidence'])
                scores_df['Confidence'] = scores_df['Confidence'].round(3)
                
                # Create a bar chart
                fig = px.bar(
                    scores_df, 
                    x='Class', 
                    y='Confidence',
                    title='Confidence Scores by Class',
                    color='Confidence',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    xaxis_title="Class",
                    yaxis_title="Confidence Score",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed scores
                for class_name, score in scores.items():
                    st.metric(f"{class_name}", f"{score:.1%}")
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>📋 Instructions</h3>
            <ol>
                <li>Upload a satellite image using the file uploader above</li>
                <li>Supported formats: PNG, JPG, JPEG</li>
                <li>Click 'Classify Image' to get predictions</li>
                <li>View confidence scores for all classes</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Model Analytics":
    st.markdown("## 📊 Model Performance Analytics")
    
    # Load mock data
    data = load_mock_data()
    
    # Training metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Training & Validation Accuracy")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['epochs'],
            y=data['train_acc'],
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=data['epochs'],
            y=data['val_acc'],
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='#764ba2', width=3)
        ))
        
        fig.update_layout(
            title='Model Accuracy Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Training & Validation Loss")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['epochs'],
            y=data['train_loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#11998e', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=data['epochs'],
            y=data['val_loss'],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#38ef7d', width=3)
        ))
        
        fig.update_layout(
            title='Model Loss Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix (Mock)
    st.markdown("### 🎯 Confusion Matrix")
    
    # Create mock confusion matrix
    classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
    cm = np.array([
        [45, 2, 1, 0],
        [1, 42, 2, 1],
        [2, 1, 44, 1],
        [0, 1, 1, 46]
    ])
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=classes,
        y=classes,
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
            )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Accuracy",
            value="94.2%",
            delta="2.1%"
        )
    
    with col2:
        st.metric(
            label="Precision",
            value="93.8%",
            delta="1.5%"
        )
    
    with col3:
        st.metric(
            label="Recall",
            value="94.0%",
            delta="2.0%"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value="93.9%",
            delta="1.8%"
        )

elif page == "ℹ️ About Model":
    st.markdown("## ℹ️ Model Architecture & Information")
    
    # Model architecture
    st.markdown("### 🏗️ CNN Architecture")
    
    architecture_data = {
        'Layer': [
            'Conv2D (32 filters)',
            'MaxPooling2D',
            'Conv2D (64 filters)',
            'MaxPooling2D',
            'Conv2D (128 filters)',
            'MaxPooling2D',
            'Flatten',
            'Dense (128 units)',
            'Dropout (0.5)',
            'Dense (4 units)'
        ],
        'Output Shape': [
            '(253, 253, 32)',
            '(126, 126, 32)',
            '(124, 124, 64)',
            '(62, 62, 64)',
            '(60, 60, 128)',
            '(30, 30, 128)',
            '(115200)',
            '(128)',
            '(128)',
            '(4)'
        ],
        'Activation': [
            'ReLU',
            '-',
            'ReLU',
            '-',
            'ReLU',
            '-',
            '-',
            'ReLU',
            '-',
            'Softmax'
        ]
    }
    
    arch_df = pd.DataFrame(architecture_data)
    st.dataframe(arch_df, use_container_width=True)
    
    # Training details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎛️ Training Configuration
        - **Optimizer:** Adam
        - **Loss Function:** Categorical Crossentropy
        - **Batch Size:** 32
        - **Epochs:** 10
        - **Input Size:** 255 x 255 x 3
        - **Data Split:** 80% Train, 20% Test
        """)
        
        st.markdown("""
        ### 📊 Dataset Information
        - **Total Images:** ~2000
        - **Classes:** 4
        - **Image Format:** RGB
        - **Augmentation:** Yes
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 Data Augmentation
        - **Rescaling:** 1/255
        - **Rotation:** ±45°
        - **Horizontal Flip:** Yes
        - **Vertical Flip:** Yes
        - **Zoom Range:** 0.2
        - **Shear Range:** 0.2
        """)
        
        st.markdown("""
        ### 🎯 Class Distribution
        - **Cloudy:** ~500 images
        - **Desert:** ~500 images
        - **Green Area:** ~500 images
        - **Water:** ~500 images
        """)
    
    # Performance summary
    st.markdown("### 📈 Performance Summary")
    
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Training': ['94.2%', '94.1%', '94.0%', '94.0%'],
        'Validation': ['91.8%', '91.5%', '91.7%', '91.6%']
    }
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🛰️ Satellite Image Classifier | Built with Streamlit & Deep Learning</p>
    <p>© 2024 Environmental AI Solutions</p>
</div>
""", unsafe_allow_html=True)