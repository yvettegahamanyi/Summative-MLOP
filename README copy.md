# Waste Classification MLOps Pipeline

A complete MLOps pipeline for waste classification using EfficientNetB0, PostgreSQL, and FastAPI.

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
│   └── database.py          # PostgreSQL database utilities
├── models/                  # Saved models (created after training)
│   ├── waste_classifier_final.tf/
│   └── model_config.json
└── requirements.txt
```

## Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up PostgreSQL database:**

   - Create a PostgreSQL database on Render (or any PostgreSQL host)
   - Set the `DATABASE_URL` environment variable:

   ```bash
   export DATABASE_URL="postgresql://user:password@host:port/database"
   ```

3. **Train the model:**
   - Run the notebook `notebook/waste-classification.ipynb` in Google Colab
   - The model will be saved to `models/waste_classifier_final.tf`

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

#### 1. Predict Single Image

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

#### 2. Predict Multiple Images

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

#### 3. Upload Image for Retraining

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@image.jpg" \
  -F "class_name=Plastic"
```

#### 4. Upload Multiple Images for Retraining

```bash
curl -X POST "http://localhost:8000/upload/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "class_names=Plastic,Paper"
```

#### 5. Trigger Model Retraining

```bash
curl -X POST "http://localhost:8000/retrain"
```

#### 6. Get Database Statistics

```bash
curl "http://localhost:8000/stats"
```

#### 7. Get Class Names

```bash
curl "http://localhost:8000/classes"
```

#### 8. Health Check

```bash
curl "http://localhost:8000/health"
```

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

## Workflow

1. **Initial Training**: Train model using the notebook
2. **Prediction**: Use API to classify waste images
3. **Data Collection**: Upload new images via API to database
4. **Retraining**: Trigger retraining when enough new data is available
5. **Model Update**: New model automatically replaces the old one

## Database Schema

The PostgreSQL database has two main tables:

- `training_images`: Stores uploaded images with metadata
- `training_runs`: Tracks retraining runs and metrics

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (required)
- `PORT`: Server port (default: 8000)

## Notes

- Models are saved in TensorFlow SavedModel format (`.tf` directories)
- The model loads from `models/waste_classifier_final.tf` by default
- Retraining uses data from the PostgreSQL database
- All images are validated before being saved to the database
