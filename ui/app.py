"""
Waste Classification UI
A comprehensive interface for model monitoring, prediction, and retraining.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://0.0.0.0:8000")

# Page configuration
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def check_backend_health():
    """Check if backend is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


@st.cache_data(ttl=30)
def get_classes():
    """Get list of classes from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/classes", timeout=5)
        if response.status_code == 200:
            return response.json().get("classes", [])
    except:
        pass
    return []


@st.cache_data(ttl=10)
def get_training_runs(limit=10, force_fetch=False):
    """Get training runs from backend."""
    # Check if retraining is in progress and avoid API calls unless forced
    if not force_fetch and hasattr(st.session_state, 'retraining_in_progress') and st.session_state.retraining_in_progress:
        return []
    
    try:
        response = requests.get(f"{BACKEND_URL}/retrain/runs?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json().get("runs", [])
    except:
        pass
    return []


def get_model_uptime():
    """Calculate model uptime based on health checks."""
    # In a real scenario, you'd track this over time
    # For demo, we'll use current health status
    health = check_backend_health()
    if health and health.get("status") == "healthy":
        return "Online", "🟢"
    return "Offline", "🔴"


def main():
    """Main application."""
    st.markdown('<h1 class="main-header">♻️ Waste Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Prediction", "Upload Data", "Retraining", "Visualizations"]
    )
    
    # Get backend status
    health = check_backend_health()
    classes = get_classes()
    
    if page == "Dashboard":
        show_dashboard(health, classes)
    elif page == "Prediction":
        show_prediction(health, classes)
    elif page == "Upload Data":
        show_upload_data(health, classes)
    elif page == "Retraining":
        show_retraining(health)
    elif page == "Visualizations":
        show_visualizations(health, classes)


def show_dashboard(health, classes):
    """Dashboard page showing model uptime and status."""
    st.header("📊 Dashboard")
    
    # Model Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status, icon = get_model_uptime()
        st.metric("Model Status", f"{icon} {status}")
    
    with col2:
        if health:
            num_classes = health.get("num_classes", 0)
            st.metric("Number of Classes", num_classes)
        else:
            st.metric("Number of Classes", "N/A")
    
    with col3:
        if health:
            img_size = health.get("image_size", "N/A")
            st.metric("Image Size", str(img_size))
        else:
            st.metric("Image Size", "N/A")
    
    with col4:
        if health and health.get("status") == "healthy":
            uptime_status = "🟢 Online"
            st.metric("Uptime", uptime_status)
        else:
            uptime_status = "🔴 Offline"
            st.metric("Uptime", uptime_status)
    
    st.divider()
    
    # Health Details
    st.subheader("System Health")
    if health:
        if health.get("status") == "healthy":
            st.success(f"✅ Backend is healthy. Model loaded: {health.get('model_loaded', False)}")
            st.json(health)
        else:
            st.error(f"❌ Backend is unhealthy: {health.get('message', 'Unknown error')}")
    else:
        st.error("❌ Cannot connect to backend. Please check if the server is running.")
        st.info(f"Backend URL: {BACKEND_URL}")
    
    # Recent Training Runs
    st.subheader("Recent Training Runs")
    
    # Check if retraining is in progress
    retraining_in_progress = hasattr(st.session_state, 'retraining_in_progress') and st.session_state.retraining_in_progress
    
    if retraining_in_progress:
        st.info("⏳ Retraining is currently in progress. Training run data will be updated after completion.")
    else:
        runs = get_training_runs(limit=5)
        if runs:
            runs_df = pd.DataFrame(runs)
            st.dataframe(runs_df[['id', 'started_at', 'completed_at', 'status']], use_container_width=True)
        else:
            st.info("No training runs found.")


def show_prediction(health, classes):
    """Prediction page for uploading images and getting predictions."""
    st.header("🔮 Prediction")
    
    if not health or health.get("status") != "healthy":
        st.error("❌ Backend is not available. Please check the server status.")
        return
    
    st.subheader("Upload Image for Classification")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
        help="Upload an image to classify"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Predict", type="primary"):
                with st.spinner("Processing prediction..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("✅ Prediction Complete!")
                            
                            # Display prediction
                            predicted_label = result.get("label", "Unknown")
                            confidence = result.get("confidence", 0)
                            
                            st.metric("Predicted Class", predicted_label)
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Show all probabilities
                            probabilities = result.get("probabilities", {})
                            if probabilities:
                                st.subheader("Class Probabilities")
                                prob_df = pd.DataFrame([
                                    {"Class": k, "Probability": v}
                                    for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                                ])
                                
                                # Bar chart
                                fig = px.bar(
                                    prob_df,
                                    x="Class",
                                    y="Probability",
                                    title="Prediction Probabilities",
                                    color="Probability",
                                    color_continuous_scale="Blues"
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Table
                                st.dataframe(prob_df, use_container_width=True)
                        else:
                            st.error(f"❌ Prediction failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


def show_upload_data(health, classes):
    """Upload data page for adding training images."""
    st.header("📤 Upload Training Data")
    
    if not health:
        st.error("❌ Backend is not available. Please check the server status.")
        return
    
    # st.subheader("Single Image Upload")
    
    # uploaded_file = st.file_uploader(
    #     "Choose an image file",
    #     type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
    #     key="single_upload"
    # )
    
    # if uploaded_file:
    #     st.image(uploaded_file, caption="Image to Upload", width=300)
        
    #     selected_class = st.selectbox(
    #         "Select Class",
    #         classes,
    #         key="single_class"
    #     )
        
    #     if st.button("📤 Upload Image", type="primary"):
    #         with st.spinner("Uploading image..."):
    #             try:
    #                 uploaded_file.seek(0)
    #                 files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    #                 data = {'class_name': selected_class}
    #                 response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data, timeout=30)
                    
    #                 if response.status_code == 200:
    #                     result = response.json()
    #                     st.success(f"✅ Image uploaded successfully! ID: {result.get('image_id')}")
    #                 else:
    #                     st.error(f"❌ Upload failed: {response.text}")
    #             except Exception as e:
    #                 st.error(f"❌ Error: {str(e)}")
    
    # st.divider()
    
    st.subheader("Batch Image Upload")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} files selected**")
        
        # Create a dictionary to store class assignments
        if 'class_assignments' not in st.session_state:
            st.session_state.class_assignments = {}
        
        # Display files with class selectors
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files):
            col_idx = idx % 3
            with cols[col_idx]:
                st.image(file, caption=file.name, use_container_width=True)
                selected_class = st.selectbox(
                    f"Class for {file.name}",
                    classes,
                    key=f"class_{idx}",
                    index=0
                )
                st.session_state.class_assignments[file.name] = selected_class
        
        if st.button("📤 Upload All Images", type="primary"):
            with st.spinner("Uploading images..."):
                try:
                    # Build class names list
                    class_names_list = []
                    for file in uploaded_files:
                        class_name = st.session_state.class_assignments.get(file.name, classes[0])
                        class_names_list.append(class_name)
                    
                    # Prepare files - reset file pointers
                    files_prep = []
                    for file in uploaded_files:
                        file.seek(0)
                        files_prep.append(('files', (file.name, file, file.type)))
                    
                    # Prepare query parameter - comma-separated string
                    class_names_str = ','.join(class_names_list)
                    
                    # Make request with query parameter
                    response = requests.post(
                        f"{BACKEND_URL}/upload/batch",
                        files=files_prep,
                        params={'class_names': class_names_str},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Successfully uploaded {result.get('total', 0)} images!")
                        st.json(result)
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def show_retraining(health):
    """Retraining page for triggering and monitoring retraining."""
    st.header("🔄 Model Retraining")
    
    if not health:
        st.error("❌ Backend is not available. Please check the server status.")
        return
    
    # Initialize session state for retraining
    if 'retraining_in_progress' not in st.session_state:
        st.session_state.retraining_in_progress = False
    if 'last_retraining_start' not in st.session_state:
        st.session_state.last_retraining_start = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trigger Retraining")
        st.info("""
        This will retrain the model using all images in the database.
        The process runs in the background and takes approximately 5-10 minutes.
        """)
        
        # Show different UI based on retraining status
        if st.session_state.retraining_in_progress:
            st.warning("⏳ Retraining is currently in progress. Please wait...")
            if st.button("🛑 Mark as Complete", help="Click if retraining finished"):
                st.session_state.retraining_in_progress = False
                st.cache_data.clear()
                st.rerun()
        else:
            if st.button("🚀 Start Retraining", type="primary"):
                with st.spinner("Starting retraining..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/retrain", timeout=10)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result.get('message')}")
                            st.balloons()
                            # Mark retraining as in progress
                            st.session_state.retraining_in_progress = True
                            st.session_state.last_retraining_start = datetime.now()
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to start retraining: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Training Status")
        
        # Only show refresh button if not retraining
        if not st.session_state.retraining_in_progress:
            if st.button("🔄 Refresh Status"):
                st.cache_data.clear()
        else:
            st.info("🔄 Auto-refresh disabled during retraining to prevent interruption")
            if st.session_state.last_retraining_start:
                elapsed = datetime.now() - st.session_state.last_retraining_start
                st.write(f"⏱️ Retraining started: {elapsed.seconds // 60}m {elapsed.seconds % 60}s ago")
        
        # Only fetch training runs if not currently retraining
        if not st.session_state.retraining_in_progress:
            runs = get_training_runs(limit=10)
        else:
            # Periodically check if retraining has completed
            if st.button("🔍 Check if Retraining Completed", help="Check backend for completion status"):
                with st.spinner("Checking retraining status..."):
                    try:
                        # Force fetch to check status
                        recent_runs = get_training_runs(limit=1, force_fetch=True)
                        if recent_runs:
                            latest_run = recent_runs[0]
                            if latest_run.get('status') in ['completed', 'failed']:
                                st.session_state.retraining_in_progress = False
                                st.success("✅ Retraining process detected as completed!")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.info(f"⏳ Retraining still in progress. Status: {latest_run.get('status', 'unknown')}")
                    except Exception as e:
                        st.warning(f"Could not check status: {str(e)}")
            
            runs = []
            st.info("📊 Training run data will be available after retraining completes")
        if runs:
            # Filter for most recent
            latest_run = runs[0] if runs else None
            
            if latest_run:
                status = latest_run.get('status', 'unknown')
                if status == 'completed':
                    st.success(f"✅ Status: {status.upper()}")
                elif status == 'failed':
                    st.error(f"❌ Status: {status.upper()}")
                elif status == 'started':
                    st.warning(f"⏳ Status: {status.upper()}")
                else:
                    st.info(f"ℹ️ Status: {status.upper()}")
                
                st.write(f"**Started:** {latest_run.get('started_at', 'N/A')}")
                st.write(f"**Completed:** {latest_run.get('completed_at', 'In Progress')}")
                
                metrics = latest_run.get('metrics')
                if metrics:
                    st.subheader("Training Metrics")
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df, use_container_width=True)
        
        st.subheader("Recent Training Runs")
        if runs:
            runs_df = pd.DataFrame(runs)
            st.dataframe(
                runs_df[['id', 'started_at', 'completed_at', 'status']],
                use_container_width=True
            )
        else:
            st.info("No training runs found.")


def get_class_descriptions():
    """Get descriptions for each waste class."""
    return {
        "Cardboard": "Cardboard boxes, packaging materials, and corrugated cardboard used for shipping and storage.",
        "Food Organics": "Food waste including fruit peels, vegetable scraps, leftover food, and organic kitchen waste.",
        "Glass": "Glass bottles, jars, containers, and other glass products that can be recycled.",
        "Metal": "Metal cans, aluminum foil, metal containers, and other metallic waste items.",
        "Miscellaneous Trash": "General waste items that don't fit into other categories, including mixed materials and non-recyclable items.",
        "Paper": "Newspapers, magazines, office paper, notebooks, and other paper-based materials.",
        "Plastic": "Plastic bottles, containers, bags, packaging materials, and other plastic products.",
        "Textile Trash": "Clothing, fabrics, textiles, and fabric-based waste materials.",
        "Vegetation": "Garden waste, leaves, branches, grass clippings, and other plant-based organic materials."
    }


@st.cache_data(ttl=300)
def get_example_image_for_class(class_name):
    """Get an example image for a class from the backend."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/classes/{class_name}/example",
            timeout=5
        )
        if response.status_code == 200:
            return response.content
    except:
        pass
    return None


def show_visualizations(health, classes):
    """Visualizations page showing data insights."""
    st.header("📈 Data Visualizations")
    
    if not health:
        st.error("❌ Backend is not available. Please check the server status.")
        return
    
    # Class Information Section
    st.subheader("📚 Waste Classification Classes")
    st.markdown("""
    The model can classify waste into **9 distinct categories**. Each category represents 
    a different type of waste material that requires specific handling and recycling processes.
    """)
    
    class_descriptions = get_class_descriptions()
    
    # Display classes in an organized grid with descriptions
    # Use tabs or expandable sections for better organization
    tab1, tab2 = st.tabs(["All Classes Overview", "Class Details"])
    
    with tab1:
        st.markdown("### Quick Overview")
        # Create a summary table
        class_data = []
        for class_name in classes:
            description = class_descriptions.get(class_name, "No description available.")
            # Truncate description for table
            short_desc = description[:60] + "..." if len(description) > 60 else description
            class_data.append({
                "Class Name": class_name,
                "Description": short_desc
            })
        
        class_df = pd.DataFrame(class_data)
        st.dataframe(class_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Detailed Class Information")
        st.markdown("Click on a class to see detailed information and examples.")
        
        # Display classes in a grid with expandable sections
        num_cols = 3
        class_rows = [classes[i:i + num_cols] for i in range(0, len(classes), num_cols)]
        
        for row_idx, row_classes in enumerate(class_rows):
            cols = st.columns(num_cols)
            for idx, class_name in enumerate(row_classes):
                with cols[idx]:
                    # Create an expandable card for each class
                    with st.expander(f"**{class_name}**", expanded=False):
                        # Try to get example image
                        example_image = get_example_image_for_class(class_name)
                        
                        if example_image:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(example_image))
                            st.image(
                                img,
                                caption=f"Example: {class_name}",
                                use_container_width=True
                            )
                        else:
                            st.caption(
                                "📷 No example image available. "
                                "Upload images to see examples here."
                            )
                        
                        description = class_descriptions.get(
                            class_name, "No description available."
                        )
                        st.write(description)
                        
                        # Icon/emoji for each class type
                        class_icons = {
                            "Cardboard": "📦",
                            "Food Organics": "🍎",
                            "Glass": "🍶",
                            "Metal": "🥫",
                            "Miscellaneous Trash": "🗑️",
                            "Paper": "📄",
                            "Plastic": "🧴",
                            "Textile Trash": "👕",
                            "Vegetation": "🌿"
                        }
                        icon = class_icons.get(class_name, "📋")
                        st.markdown(f"**Category Icon:** {icon}")
                        
                        # Recycling info
                        recyclable_classes = [
                            "Cardboard", "Glass", "Metal", "Paper", "Plastic"
                        ]
                        if class_name in recyclable_classes:
                            st.success("♻️ Recyclable")
                        elif class_name in ["Food Organics", "Vegetation"]:
                            st.info("🌱 Compostable")
                        else:
                            st.warning("⚠️ Special handling required")


if __name__ == "__main__":
    main()

