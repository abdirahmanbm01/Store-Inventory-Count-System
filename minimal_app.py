import streamlit as st
import pandas as pd
from PIL import Image
import io
from datetime import datetime, timezone
import os

# Simple test version - no YOLO yet
st.set_page_config(page_title="Retail Counter Test", page_icon="👜")

st.title("👜 Retail Display Counter (Test Version)")
st.caption("Testing the basic UI before adding AI detection")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    store = st.text_input("Store", value="Demo Store")
    wall_id = st.text_input("Wall/Display ID", value="Wall-A")
    session = st.text_input("Session Tag", value="")
    
    st.divider()
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.35)
    iou = st.slider("NMS IoU", 0.3, 0.8, 0.45)
    
    categories = ["Bag", "Shoe", "SLG", "Belt", "Accessory", "Unknown"]
    auto_mode = st.checkbox("Auto-count mode", value=True)
    auto_category = st.selectbox("Auto category", categories, index=0)
    
    output_folder = st.text_input("Output folder", value="outputs")
    
    st.info("📱 Tip: One photo per wall; style IDs will be added later")

# Main area
st.header("Step 1: Upload Images")
uploaded_files = st.file_uploader(
    "Choose images", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    all_detections = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"Image {i+1}: {uploaded_file.name}")
        
        # Show the image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        # Simulate some detections for testing
        fake_detections = [
            {"category": auto_category, "confidence": 0.85, "x1": 100, "y1": 100, "x2": 200, "y2": 200},
            {"category": auto_category, "confidence": 0.72, "x1": 300, "y1": 150, "x2": 400, "y2": 250},
        ]
        
        st.success(f"✅ Detections found: {len(fake_detections)}")
        
        if auto_mode:
            st.info(f"🤖 Auto-count mode ON — every detection counted as: {auto_category}")
        
        # Show per-photo counts
        df_counts = pd.DataFrame(fake_detections).groupby('category').size().reset_index(name='count')
        st.subheader("Per-photo counts:")
        st.dataframe(df_counts, use_container_width=True)
        
        # Add to session data
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for det in fake_detections:
            all_detections.append({
                'timestamp_utc': timestamp,
                'store': store,
                'wall_id': wall_id,
                'session': session,
                'file': uploaded_file.name,
                'detected_class': 'handbag',  # fake for now
                'confidence': det['confidence'],
                'category': det['category'],
                'x1': det['x1'], 'y1': det['y1'], 'x2': det['x2'], 'y2': det['y2']
            })

    # Step 2: Export Results
    if all_detections:
        st.header("Step 2: Export Results")
        
        # Session rollup
        df_all = pd.DataFrame(all_detections)
        rollup = df_all.groupby(['store', 'wall_id', 'session', 'category']).size().reset_index(name='count')
        
        st.subheader("Session Roll-up:")
        st.dataframe(rollup, use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv_detections = df_all.to_csv(index=False)
            st.download_button(
                "📄 Download Per-Detection CSV",
                csv_detections,
                file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_rollup = rollup.to_csv(index=False)
            st.download_button(
                "📊 Download Roll-up CSV", 
                csv_rollup,
                file_name=f"rollup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Save to outputs folder
        if st.button("💾 Save CSVs to outputs folder"):
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            det_file = f"{output_folder}/{timestamp}_detections.csv"
            rollup_file = f"{output_folder}/{timestamp}_rollup.csv"
            
            df_all.to_csv(det_file, index=False)
            rollup.to_csv(rollup_file, index=False)
            
            st.success(f"✅ Saved to {det_file} and {rollup_file}")

else:
    st.info("👆 Upload some images to get started!")

# Footer
st.divider()
st.markdown("🚧 **Test Version** - YOLO detection will be added next")