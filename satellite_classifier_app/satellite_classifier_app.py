import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import pickle
import os

# Set page config
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🛰️",
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
        margin-bottom: 0.5rem;
        font-size: 3rem;
    }
    
    .main-header p {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.2rem;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px 10px 0 0;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .class-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Mock model class for demonstration
class MockModel:
    def __init__(self):
        self.classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
        self.class_colors = ['#87CEEB', '#DEB887', '#90EE90', '#4682B4']
        
    def predict(self, image_array):
        # Generate realistic-looking predictions
        np.random.seed(42)  # For consistent demo results
        
        # Simulate different prediction patterns based on image characteristics
        probabilities = np.random.dirichlet(np.ones(4), size=1)[0]
        
        # Add some randomness but keep it realistic
        probabilities = probabilities * 0.8 + np.random.random(4) * 0.2
        probabilities = probabilities / probabilities.sum()
        
        return probabilities.reshape(1, -1)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Helper functions
def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize((255, 255))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_tensorflow_model(model_path):
    """Load TensorFlow model - this will work when TensorFlow is installed"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except ImportError:
        st.error("TensorFlow not installed. Please install with: pip install tensorflow")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>AI-Powered Land Cover Classification System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Model Configuration")
    
    # Model loading section
    st.markdown("#### Load Model")
    
    # Demo mode toggle
    demo_mode = st.checkbox("🎮 Demo Mode", help="Use mock predictions for demonstration")
    
    if demo_mode:
        st.session_state.model = MockModel()
        st.session_state.demo_mode = True
        st.success("✅ Demo mode activated!")
        st.info("💡 This uses simulated predictions. Upload your .h5 model file to use real predictions.")
    else:
        model_file = st.file_uploader(
            "Upload your trained model (.h5 file)",
            type=['h5'],
            help="Upload the Modelenv.v1.h5 file you trained"
        )
        
        if model_file is not None:
            try:
                # Save uploaded file temporarily
                with open("temp_model.h5", "wb") as f:
                    f.write(model_file.getvalue())
                
                # Load model
                model = load_tensorflow_model("temp_model.h5")
                if model is not None:
                    st.session_state.model = model
                    st.session_state.demo_mode = False
                    st.success("✅ Model loaded successfully!")
                
                # Clean up temp file
                if os.path.exists("temp_model.h5"):
                    os.remove("temp_model.h5")
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    st.markdown("---")
    
    # Installation guide
    if not demo_mode:
        st.markdown("#### 🚀 Installation Guide")
        st.markdown("""
        **If TensorFlow installation fails:**
        ```bash
        # Option 1: CPU version
        pip install tensorflow-cpu
        
        # Option 2: Specific version
        pip install tensorflow==2.12.0
        
        # Option 3: Use conda
        conda install tensorflow
        ```
        """)
    
    st.markdown("---")
    
    # Class information
    st.markdown("#### 📊 Classification Classes")
    classes = ['Cloudy', 'Desert', 'Green Area', 'Water']
    class_colors = ['#87CEEB', '#DEB887', '#90EE90', '#4682B4']
    
    for i, (cls, color) in enumerate(zip(classes, class_colors)):
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <div style="width: 20px; height: 20px; background: {color}; border-radius: 50%; margin-right: 10px;"></div>
            <span>{cls}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### 📈 Session Statistics")
    if st.session_state.prediction_history:
        total_predictions = len(st.session_state.prediction_history)
        avg_confidence = np.mean([pred['confidence'] for pred in st.session_state.prediction_history])
        
        st.metric("Total Predictions", total_predictions)
        st.metric("Average Confidence", f"{avg_confidence:.2f}%")
        
        # Class distribution
        class_counts = {}
        for pred in st.session_state.prediction_history:
            cls = pred['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        if class_counts:
            st.markdown("**Class Distribution:**")
            for cls, count in class_counts.items():
                st.write(f"• {cls}: {count}")
    
    # Clear history button
    if st.session_state.prediction_history:
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# Main content area
if st.session_state.model is None:
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Getting Started</h3>
        <p>To use this satellite image classifier, you have two options:</p>
        <ol>
            <li><strong>Demo Mode:</strong> Enable demo mode in the sidebar to see the interface with simulated predictions</li>
            <li><strong>Real Model:</strong> Upload your trained model file (.h5) to use real predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # TensorFlow installation help
    st.markdown("""
    <div class="info-box">
        <h3>⚙️ TensorFlow Installation</h3>
        <p>If you're having trouble installing TensorFlow, try these commands:</p>
        <pre>
# For CPU version (recommended for most users)
pip install tensorflow-cpu

# Or upgrade pip first
python -m pip install --upgrade pip
pip install tensorflow

# Using conda
conda install tensorflow
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Show demo information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Real-time Classification**: Instant prediction results
        - **Confidence Scores**: Detailed probability analysis
        - **Batch Processing**: Upload multiple images at once
        - **Visualization**: Interactive charts and heatmaps
        - **History Tracking**: Monitor prediction performance
        - **Demo Mode**: Test the interface without a model
        """)
    
    with col2:
        st.markdown("### 🔍 Supported Classes")
        st.markdown("""
        - **☁️ Cloudy**: Cloud-covered areas
        - **🏜️ Desert**: Arid and desert regions
        - **🌱 Green Area**: Vegetation and forests
        - **💧 Water**: Water bodies and oceans
        """)

else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Analytics", "📈 Visualizations", "🎯 Batch Process"])
    
    with tab1:
        st.markdown("### 🖼️ Image Classification")
        
        if st.session_state.demo_mode:
            st.info("🎮 Demo mode active - using simulated predictions")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a satellite image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a satellite image for classification"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image_pil = Image.open(uploaded_file)
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)
                
                # Make prediction
                if st.button("🔍 Classify Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        try:
                            processed_img = preprocess_image(image_pil)
                            prediction = st.session_state.model.predict(processed_img)
                            
                            # Get prediction results
                            classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
                            predicted_class_idx = np.argmax(prediction[0])
                            predicted_class = classes[predicted_class_idx]
                            confidence = prediction[0][predicted_class_idx] * 100
                            
                            # Store in history
                            st.session_state.prediction_history.append({
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'probabilities': prediction[0]
                            })
                            
                            # Display results in next column
                            with col2:
                                st.markdown("#### 🎯 Prediction Results")
                                
                                # Main prediction card
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <h2>{predicted_class.replace('_', ' ')}</h2>
                                    <h3>{confidence:.1f}% Confidence</h3>
                                    {"<p>🎮 Simulated Result</p>" if st.session_state.demo_mode else ""}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Probability distribution
                                st.markdown("#### 📊 Probability Distribution")
                                prob_data = pd.DataFrame({
                                    'Class': [c.replace('_', ' ') for c in classes],
                                    'Probability': prediction[0] * 100
                                })
                                
                                fig = px.bar(
                                    prob_data,
                                    x='Probability',
                                    y='Class',
                                    orientation='h',
                                    color='Probability',
                                    color_continuous_scale='viridis',
                                    title="Classification Probabilities"
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
        
        with col2:
            if uploaded_file is None:
                st.markdown("""
                <div class="upload-box">
                    <h3>👆 Upload an image to get started</h3>
                    <p>Supported formats: JPG, JPEG, PNG</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Model Analytics")
        
        if st.session_state.prediction_history:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_preds = len(st.session_state.prediction_history)
                st.metric("Total Predictions", total_preds)
            
            with col2:
                avg_conf = np.mean([p['confidence'] for p in st.session_state.prediction_history])
                st.metric("Average Confidence", f"{avg_conf:.1f}%")
            
            with col3:
                # Most predicted class
                class_counts = {}
                for pred in st.session_state.prediction_history:
                    cls = pred['predicted_class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                if class_counts:
                    most_common = max(class_counts.items(), key=lambda x: x[1])
                    st.metric("Most Predicted", most_common[0])
            
            # Detailed analytics
            st.markdown("#### 📈 Prediction Analysis")
            
            # Create DataFrame from history
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Class distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                if class_counts:
                    fig = px.pie(
                        values=list(class_counts.values()),
                        names=list(class_counts.keys()),
                        title="Class Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(
                    history_df,
                    x='confidence',
                    nbins=20,
                    title="Confidence Score Distribution"
                )
                fig.update_layout(
                    xaxis_title="Confidence (%)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions table
            st.markdown("#### 📋 Recent Predictions")
            recent_df = pd.DataFrame(st.session_state.prediction_history[-10:])
            recent_df['Confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1f}%")
            recent_df['Predicted Class'] = recent_df['predicted_class'].str.replace('_', ' ')
            
            st.dataframe(
                recent_df[['Predicted Class', 'Confidence']].iloc[::-1],
                use_container_width=True
            )
            
        else:
            st.info("📊 No predictions yet. Upload and classify some images to see analytics!")
    
    with tab3:
        st.markdown("### 📈 Advanced Visualizations")
        
        if st.session_state.prediction_history:
            # Confidence trends
            st.markdown("#### 📊 Confidence Trends")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['Prediction Number'] = range(1, len(history_df) + 1)
            
            fig = px.line(
                history_df,
                x='Prediction Number',
                y='confidence',
                color='predicted_class',
                title="Confidence Trends Over Time",
                labels={'confidence': 'Confidence (%)', 'predicted_class': 'Class'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability heatmap
            st.markdown("#### 🔥 Probability Heatmap")
            
            prob_matrix = np.array([pred['probabilities'] for pred in st.session_state.prediction_history])
            classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
            
            fig = px.imshow(
                prob_matrix.T,
                labels=dict(x="Prediction Number", y="Class", color="Probability"),
                y=[c.replace('_', ' ') for c in classes],
                title="Probability Heatmap Across Predictions"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### 📋 Statistical Summary")
            
            summary_stats = pd.DataFrame({
                'Metric': ['Mean Confidence', 'Median Confidence', 'Std Confidence', 'Min Confidence', 'Max Confidence'],
                'Value': [
                    f"{np.mean([p['confidence'] for p in st.session_state.prediction_history]):.2f}%",
                    f"{np.median([p['confidence'] for p in st.session_state.prediction_history]):.2f}%",
                    f"{np.std([p['confidence'] for p in st.session_state.prediction_history]):.2f}%",
                    f"{np.min([p['confidence'] for p in st.session_state.prediction_history]):.2f}%",
                    f"{np.max([p['confidence'] for p in st.session_state.prediction_history]):.2f}%"
                ]
            })
            
            st.dataframe(summary_stats, use_container_width=True, hide_index=True)
            
        else:
            st.info("📈 No data available for visualization. Make some predictions first!")
    
    with tab4:
        st.markdown("### 🎯 Batch Processing")
        
        if st.session_state.demo_mode:
            st.info("🎮 Demo mode active - batch processing will use simulated predictions")
        
        st.markdown("#### 📁 Upload Multiple Images")
        batch_files = st.file_uploader(
            "Choose multiple satellite images...",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch classification"
        )
        
        if batch_files:
            st.markdown(f"**{len(batch_files)} images uploaded**")
            
            if st.button("🚀 Process Batch", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for i, file in enumerate(batch_files):
                    # Update progress
                    progress_bar.progress((i + 1) / len(batch_files))
                    
                    try:
                        # Process image
                        image_pil = Image.open(file)
                        processed_img = preprocess_image(image_pil)
                        
                        # Get prediction (works with both real and mock models)
                        prediction = st.session_state.model.predict(processed_img)
                        
                        # Get results
                        classes = ['Cloudy', 'Desert', 'Green_Area', 'Water']
                        predicted_class_idx = np.argmax(prediction[0])
                        predicted_class = classes[predicted_class_idx]
                        confidence = prediction[0][predicted_class_idx] * 100
                        
                        results.append({
                            'Filename': file.name,
                            'Predicted Class': predicted_class.replace('_', ' '),
                            'Confidence': f"{confidence:.1f}%",
                            'Raw Confidence': confidence
                        })
                        
                        # Add to history
                        st.session_state.prediction_history.append({
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'probabilities': prediction[0]
                        })
                        
                    except Exception as e:
                        results.append({
                            'Filename': file.name,
                            'Predicted Class': 'Error',
                            'Confidence': f"Error: {str(e)}",
                            'Raw Confidence': 0
                        })
                
                # Show results
                st.success(f"✅ Processed {len(batch_files)} images successfully!")
                
                # Results table
                results_df = pd.DataFrame(results)
                st.dataframe(results_df[['Filename', 'Predicted Class', 'Confidence']], use_container_width=True)
                
                # Summary statistics
                valid_results = [r for r in results if r['Predicted Class'] != 'Error']
                if valid_results:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_conf = np.mean([r['Raw Confidence'] for r in valid_results])
                        st.metric("Average Confidence", f"{avg_conf:.1f}%")
                    
                    with col2:
                        success_rate = (len(valid_results) / len(results)) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                    with col3:
                        # Most common class
                        class_counts = {}
                        for r in valid_results:
                            cls = r['Predicted Class']
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        
                        if class_counts:
                            most_common = max(class_counts.items(), key=lambda x: x[1])
                            st.metric("Most Common Class", most_common[0])
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_classification_results.csv",
                    mime="text/csv"
                )
        
        else:
            st.markdown("""
            <div class="upload-box">
                <h3>📁 Upload multiple images for batch processing</h3>
                <p>Select multiple satellite images to classify them all at once</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🛰️ Satellite Image Classifier | Built with Streamlit</p>
    <p>{"🎮 Demo Mode Active" if st.session_state.demo_mode else "🔗 Upload your trained model to start classifying!"}</p>
</div>
""", unsafe_allow_html=True)