# Waste Classification UI

A comprehensive Streamlit-based user interface for the Waste Classification ML system.

## Features

- **Dashboard**: Model uptime monitoring and system health
- **Prediction**: Upload images and get real-time predictions with confidence scores
- **Upload Data**: Single and batch image upload with class selection
- **Retraining**: Trigger and monitor model retraining
- **Visualizations**: Data insights and training metrics

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the `ui` directory:

```
BACKEND_URL=http://localhost:8000
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

### Dashboard

- View model status and uptime
- Check system health
- Monitor recent training runs

### Prediction

- Upload an image
- Get classification results with confidence scores
- View probability distribution across all classes

### Upload Data

- **Single Upload**: Upload one image and assign a class
- **Batch Upload**: Upload multiple images with individual class selectors

### Retraining

- Trigger model retraining
- Monitor training status
- View training metrics and history

### Visualizations

- Training run status distribution
- Training timeline
- Metrics visualization for completed runs

## Configuration

Set the `BACKEND_URL` in your `.env` file to point to your FastAPI backend server.
