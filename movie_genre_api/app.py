# FastAPI Web Service for Multi-Label Movie Genre Classification
# File: app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Movie Genre Prediction API",
    description="Multi-label classification API for predicting movie genres from descriptions",
    version="1.0.0"
)

# Load trained model and artifacts
try:
    model = joblib.load('genre_classifier_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    mlb = joblib.load('mlb_encoder.pkl')
    print("✓ Model and artifacts loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model, tfidf, mlb = None, None, None

# Define request/response models
class MovieDescription(BaseModel):
    description: str

    class Config:
        schema_extra = {
            "example": {
                "description": "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe."
            }
        }

class GenrePrediction(BaseModel):
    description: str
    predicted_genres: List[str]
    confidence_scores: dict

class BatchMovieDescriptions(BaseModel):
    descriptions: List[str]

class BatchPredictions(BaseModel):
    predictions: List[GenrePrediction]

# ===========================
# API ENDPOINTS
# ===========================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Genre Predictor</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }
            h1 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                color: #333;
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                font-family: inherit;
                resize: vertical;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9ff;
                border-radius: 10px;
                border-left: 5px solid #667eea;
                display: none;
            }
            .result.show {
                display: block;
                animation: slideIn 0.5s ease;
            }
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            .result h2 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }
            .genres {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 15px;
            }
            .genre-tag {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 14px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
                color: #667eea;
                font-weight: 600;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 10px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .examples {
                margin-top: 20px;
                padding: 15px;
                background: #fff9e6;
                border-radius: 10px;
                border-left: 5px solid #ffc107;
            }
            .examples h3 {
                color: #f57c00;
                margin-bottom: 10px;
                font-size: 1.2em;
            }
            .example-item {
                padding: 8px;
                margin: 5px 0;
                background: white;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.2s;
            }
            .example-item:hover {
                background: #fff3e0;
            }
            .error {
                background: #ffebee;
                border-left-color: #f44336;
                color: #c62828;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Movie Genre Predictor</h1>
            <p class="subtitle">Enter a movie description and get AI-powered genre predictions</p>

            <div class="input-group">
                <label for="description">Movie Description:</label>
                <textarea
                    id="description"
                    rows="6"
                    placeholder="Enter a movie description here... For example: 'A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe.'"
                ></textarea>
            </div>

            <button onclick="predictGenres()" id="predictBtn">Predict Genres</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing movie description...</p>
            </div>

            <div class="result" id="result">
                <h2>Predicted Genres:</h2>
                <div class="genres" id="genres"></div>
            </div>

            <div class="examples">
                <h3>💡 Try these examples:</h3>
                <div class="example-item" onclick="setExample(0)">
                    "A group of intergalactic criminals must pull together to stop a fanatical warrior..."
                </div>
                <div class="example-item" onclick="setExample(1)">
                    "A young wizard begins his magical education at a school of witchcraft and wizardry..."
                </div>
                <div class="example-item" onclick="setExample(2)">
                    "A computer hacker learns about the true nature of his reality and his role in the war..."
                </div>
            </div>
        </div>

        <script>
            const examples = [
                "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe.",
                "A young wizard begins his magical education at a school of witchcraft and wizardry where he uncovers dark secrets.",
                "A computer hacker learns about the true nature of his reality and his role in the war against its controllers."
            ];

            function setExample(index) {
                document.getElementById('description').value = examples[index];
            }

            async function predictGenres() {
                const description = document.getElementById('description').value.trim();
                const resultDiv = document.getElementById('result');
                const genresDiv = document.getElementById('genres');
                const loadingDiv = document.getElementById('loading');
                const predictBtn = document.getElementById('predictBtn');

                if (!description) {
                    alert('Please enter a movie description!');
                    return;
                }

                // Show loading
                loadingDiv.classList.add('show');
                resultDiv.classList.remove('show');
                predictBtn.disabled = true;

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ description: description })
                    });

                    if (!response.ok) {
                        throw new Error('Prediction failed');
                    }

                    const data = await response.json();

                    // Display results
                    genresDiv.innerHTML = '';
                    if (data.predicted_genres && data.predicted_genres.length > 0) {
                        data.predicted_genres.forEach(genre => {
                            const tag = document.createElement('div');
                            tag.className = 'genre-tag';
                            tag.textContent = genre;
                            genresDiv.appendChild(tag);
                        });
                    } else {
                        genresDiv.innerHTML = '<p>No genres predicted</p>';
                    }

                    resultDiv.classList.add('show');
                    resultDiv.classList.remove('error');
                } catch (error) {
                    resultDiv.classList.add('show', 'error');
                    genresDiv.innerHTML = '<p>Error: ' + error.message + '</p>';
                } finally {
                    loadingDiv.classList.remove('show');
                    predictBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or tfidf is None or mlb is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "available_genres": list(mlb.classes_)
    }

@app.post("/predict", response_model=GenrePrediction)
async def predict_genre(movie: MovieDescription):
    """Predict genres for a single movie description"""
    if model is None or tfidf is None or mlb is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Transform input
        X = tfidf.transform([movie.description])

        # Predict
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        # Get predicted genres
        predicted_genres = mlb.inverse_transform(y_pred)[0]

        # Get confidence scores (if available)
        confidence_scores = {}
        if y_pred_proba is not None:
            for idx, genre in enumerate(mlb.classes_):
                if y_pred[0][idx] == 1:
                    # Get probability for this genre
                    # Note: OneVsRestClassifier returns list of arrays
                    confidence_scores[genre] = float(y_pred_proba[idx][0][1])

        return GenrePrediction(
            description=movie.description[:100] + "..." if len(movie.description) > 100 else movie.description,
            predicted_genres=list(predicted_genres) if predicted_genres else ["Unknown"],
            confidence_scores=confidence_scores
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictions)
async def predict_genres_batch(movies: BatchMovieDescriptions):
    """Predict genres for multiple movie descriptions"""
    if model is None or tfidf is None or mlb is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for description in movies.descriptions:
            X = tfidf.transform([description])
            y_pred = model.predict(X)
            predicted_genres = mlb.inverse_transform(y_pred)[0]

            predictions.append(GenrePrediction(
                description=description[:100] + "..." if len(description) > 100 else description,
                predicted_genres=list(predicted_genres) if predicted_genres else ["Unknown"],
                confidence_scores={}
            ))

        return BatchPredictions(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/genres")
async def get_available_genres():
    """Get list of all available genre labels"""
    if mlb is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "genres": list(mlb.classes_),
        "total_count": len(mlb.classes_)
    }

# ===========================
# RUN THE APP
# ===========================

# The following section is for running the app directly, but in Colab with ngrok setup,
# we run it via a separate thread and ngrok tunnel, so this part won't be executed directly.
# if __name__ == "__main__":
#     print("="*50)
#     print("Starting Movie Genre Prediction API...")
#     print("="*50)
#     print("API Documentation: http://localhost:8000/docs")
#     print("Web Interface: http://localhost:8000")
#     print("="*50)

#     uvicorn.run(app, host="0.0.0.0", port=8000)
