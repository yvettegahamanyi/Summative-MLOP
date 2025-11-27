# Waste Classification System

Waste classification system is a platform that can classify waste into different categories Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation. The system was build using a Streamlit web user interface to allow users to use the system, it also has a FastApi backend system that exposes the api to interact with the model.

## Access the project

Backend: https://summative-mlop-1.onrender.com/docs

Frontend: https://summative-mlop-u4ckamwovu5dhcbcvaoun3.streamlit.app/

Demo: https://youtu.be/L6m5INObx8Y

## Project Structure

```
Summative-MLOP/
├── notebook/
│   └── waste-classification.ipynb  # Training notebook
├── src/
│   ├── main.py              # FastAPI application (main entry point)
│   ├── prediction.py        # Prediction logic
│   ├── preprocessing.py     # Image preprocessing and database uploads
│   ├── model.py             # Model retraining logic
│   ├── requirements.txt
│   └── database.py          # PostgreSQL database utilities
├── ui/
│   ├── app.py              # Streamlit application (main entry point)
│   └── .env                # environment variables(BACKEND_URL)
├── models/                  # Saved models (created after training)
│   ├── waste_classifier_final.tf/
│   └── model_config.json
├── locustfile.py     # Locust file to simulate flood on the system
├── README.md
├── app-requirement.txt # contains all libraries required to run the project
├── .gitignore
└── .env(with DATABASE_URL)
```

# Backend

A complete MLOps pipeline for waste classification using EfficientNetB0, PostgreSQL, and FastAPI.

## Setup Backend

1. **Install dependencies:**

```bash
pip install -r app-requirements.txt
```

2. **Set up PostgreSQL database:**

   - Create a PostgreSQL database on Render (or any PostgreSQL host)
   - Set the `DATABASE_URL` environment variable:

   ```bash
   export DATABASE_URL="postgresql://user:password@host:port/database"
   ```

3. **Train the model:**
   - Run the notebook `notebook/waste-classification.ipynb` in Google Colab
   - The model will be saved to `models/waste_classifier_final.h5`

## Usage

### Start the API Server

```bash
python src/main.py
```

Or with uvicorn:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

![alt text](assets/api_documentation.png)

## Valid Class Names

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

## Database Schema

The PostgreSQL database has two main tables:

- `training_images`: Stores uploaded images with metadata
- `training_runs`: Tracks retraining runs and metrics

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (required)
- `PORT`: Server port (default: 8000)

# Frontend

A comprehensive Streamlit-based user interface for the Waste Classification ML system.

## Features

- **Dashboard**: Model uptime monitoring and system health
- **Prediction**: Upload images and get real-time predictions with confidence scores
- **Upload Data**: Single and batch image upload with class selection
- **Retraining**: Trigger and monitor model retraining
- **Visualizations**: Data insights and training metrics

## Setup Frontend

1. **Install dependencies:**

```bash
pip install -r app-requirements.txt
```

2. **Set up backend url:**

   - Create a PostgreSQL database on Render (or any PostgreSQL host)
   - Set the `BACKEND_URL` environment variable:

   ```bash
   export DATABASE_URL="http://0.0.0.0:8000"
   ```

## Usage

### Run the application

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

- **Batch Upload**: Upload multiple images with individual class selectors

### Retraining

- Trigger model retraining
- Monitor training status
- View training metrics and history

### Visualizations

- Training run status distribution
- Training timeline
- Metrics visualization for completed runs

## Environment Variables

- `BACKEND_URL`: backend connection string (required)

## Locust Visualization

Simulate a flood of requests on waste classification backend server

![alt text](assets/locust_results.png)
