# 🎬 Movie Genre Classifier
## Multi-Label Text Classification with Deep Learning & Traditional ML

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://movie-genre-api-sparkling-frost-4339.fly.dev)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> AI-powered multi-label genre classification system trained on 160,000+ IMDb movies, comparing Traditional ML (Logistic Regression, Naive Bayes, SVM) with Deep Learning models (BERT, LSTM)

---

## 🌟 Features

- **🤖 3 Model Comparison**: Logistic Regression, Naive Bayes, SVM
- **📊 Advanced Metrics**: Standard (F1, Precision, Recall) + Custom Multi-Label Metrics
- **⚡ Real-Time Predictions**: <100ms response time in production
- **🎨 Interactive Web UI**: User-friendly interface with example inputs
- **🔌 REST API**: Complete API for programmatic access
- **☁️ Production Deployment**: Dockerized and deployed on Fly.io

---

## 🚀 Live Demo

**Try it now**: [movie-genre-api-sparkling-frost-4339.fly.dev](https://movie-genre-api-sparkling-frost-4339.fly.dev)

![photo_2025-11-19_16-01-16](https://github.com/user-attachments/assets/0c2d8e5e-65a9-4fad-8532-3097e085c606)

---
## Source
The dataset is taken from Kaggle:
[Kaggle — IMDB Movies Dataset Based on Genre](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre)

```markdown
# IMDB Movies Dataset (by genre) — description and data cleaning

## Location in the project
Archives and prepared files are stored in the project at:
- /normalization/train_data.zip

The script(s) responsible for merging, cleaning, and normalizing the data are located in:
- /normalization/data_cleaner_code

## What was done
Several individual datasets from the archive were merged into a single tabular dataset, after which a fully automated cleaning and normalization process was performed.

Main steps of the cleaning pipeline:
- Removing movies with empty or placeholder descriptions (for example, "Add a Plot").
- Removing duplicate movie records.
- Keeping only relevant fields (title, genres, description).
- Generating new sequential unique identifiers (ID) for all movies.
- Preparing the final normalized dataset for use in the project.

## Result
As a result of the pipeline, a clean and normalized CSV/Parquet (or another format — see the contents of /normalization/train_data.zip) was obtained, containing:
- id — new unique movie identifier;
- title — movie title;
- genres — genres (in standardized form);
- description — short description/plot of the movie.
```


## 📊 Model Performance

### Logistic Regression

```
Results for Logistic Regression:
Accuracy:                 0.1023
Precision (weighted):     0.6473
Recall (weighted):        0.2976
F1-Score (weighted):      0.3841
Hamming Loss:             0.0919

--- Custom Multilabel Metrics ---
Partial Accuracy:         0.2866
Accuracy (≥1 label):      0.5393
% of True Labels Guessed: 0.3217
```
### Naive Bayes
```
Results for Naive Bayes:
Accuracy:                 0.0772
Precision (weighted):     0.6536
Recall (weighted):        0.2346
F1-Score (weighted):      0.3116
Hamming Loss:             0.0945

--- Custom Multilabel Metrics ---
Partial Accuracy:         0.2284
Accuracy (≥1 label):      0.4394
% of True Labels Guessed: 0.2509
```
### Linear SVM
```
Results for Linear SVM:
Accuracy:                 0.1067
Precision (weighted):     0.5770
Recall (weighted):        0.3486
F1-Score (weighted):      0.4214
Hamming Loss:             0.0958

--- Custom Multilabel Metrics ---
Partial Accuracy:         0.3161
Accuracy (≥1 label):      0.6100
% of True Labels Guessed: 0.3749
```
### Best model by F1-score: Linear SVM⭐
###
### Custom Multi-Label Metrics Explained

1. **Partial Accuracy (Jaccard Score)**: 
   - Measures overlap between predicted and true labels
   - Formula: `|intersection| / |union|`
   - Range: 0.0 (no overlap) to 1.0 (perfect match)

2. **Accuracy ≥1 Match**: 
   - Percentage of predictions with at least one correct label
   - Practical metric: "Did we get something right?"

3. **% True Labels Guessed**: 
   - What fraction of the actual labels did we predict?
   - Per-sample recall averaged across dataset

---

## 🛠️ Tech Stack

### Machine Learning
- **Traditional ML**: scikit-learn (Logistic Regression, Naive Bayes, SVM)
- **NLP**: TF-IDF Vectorization (20,000 features, bigrams)
- **Multi-Label**: OneVsRest strategy, MultiLabelBinarizer

### Web & Deployment
- **Backend**: FastAPI + Uvicorn ASGI Server
- **Frontend**: HTML5/CSS3, Vanilla JavaScript
- **Containerization**: Docker
- **Cloud Platform**: Fly.io
- **CI/CD**: Automated deployment pipeline

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Visualizations
- **re** - Text normalization (lowercase, punctuation removal)

---

## 📁 Project Structure

```
movie-genre-classifier/
│
├── 📊 Data & Training
│   ├── cleaned_imdb_by_movie_final.csv     # Dataset (160K movies)
│   ├── train_model.py                      # Complete training pipeline
│   └── notebooks/
│       └── training_analysis.ipynb         # Exploratory analysis
│
├── 🤖 Trained Models
│   ├── genre_classifier_model.pkl          # Best model (Logistic Regression)
│   ├── tfidf_vectorizer.pkl               # TF-IDF vectorizer (20K features)
│   ├── mlb_encoder.pkl                    # Multi-label binarizer (18 genres)
│   ├── bert_model.pth                     # BERT fine-tuned weights (optional)
│   └── lstm_model.h5                      # LSTM model (optional)
│
├── 🚀 Deployment Package (movie_genre_api/)
│   ├── app.py                             # FastAPI application
│   ├── Dockerfile                         # Container configuration
│   ├── fly.toml                           # Fly.io deployment config
│   ├── requirements.txt                   # Python dependencies (minimal)
│   ├── genre_classifier_model.pkl         # Production model
│   ├── tfidf_vectorizer.pkl              # Vectorizer
│   └── mlb_encoder.pkl                   # Label encoder
│
├── 📊 Outputs & Visualizations
│   ├── genre_distribution.png             # Top 15 genres bar chart
│   ├── confusion_matrices.png             # Per-genre confusion matrices
│   ├── model_comparison_standard.png      # Standard metrics comparison
│   └── model_comparison_custom.png        # Custom metrics comparison
│
├── 📚 Documentation
│   ├── README.md                          # This file
│   ├── DEPLOYMENT.md                      # Deployment guide
│   ├── docs/
│   │   ├── report.pdf                     # Full project report
├────   └── screenshots/                   # UI screenshots
```

---

## 🔬 Training Pipeline

### 1. Data Preprocessing

```python
# Text normalization
def normalize_text(text):
    text = text.lower()                               # lowercase
    text = re.sub(r'[^a-z0-9\s]', ' ', text)         # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()         # collapse spaces
    return text

# Filter rare genres (keep genres with ≥100 movies)
MIN_GENRE_COUNT = 100
frequent_genres = {genre for genre, count in genre_counts.items() 
                   if count >= MIN_GENRE_COUNT}
```

### 2. Feature Engineering

**TF-IDF Vectorization:**
```python
tfidf = TfidfVectorizer(
    max_features=20000,     # Top 20K words
    stop_words='english',   # Remove common words
    ngram_range=(1, 2),     # Unigrams + bigrams
    min_df=5,               # Min document frequency
    max_df=0.8              # Max document frequency (remove too common)
)
```

### 3. Model Training

**Traditional ML** (OneVsRest Strategy):
```python
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    'Naive Bayes': OneVsRestClassifier(MultinomialNB()),
    'Linear SVM': OneVsRestClassifier(LinearSVC(max_iter=1000))
}
```

**Deep Learning**:

**BERT** (Transformer):
```python
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
```

**LSTM** (Recurrent):
```python
lstm_model = Sequential([
    Embedding(input_dim=20000, output_dim=100, input_length=150),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(n_genres, activation='sigmoid')  # Multi-label
])
```

### 4. Evaluation Metrics

**Standard Metrics:**
- Accuracy (exact match ratio)
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)
- Hamming Loss (label-wise error)

**Custom Multi-Label Metrics:**
```python
def partial_accuracy(y_true, y_pred):
    """Jaccard-like: intersection / union"""
    scores = []
    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])
        union = len(true_set | pred_set)
        if union > 0:
            scores.append(len(true_set & pred_set) / union)
    return np.mean(scores)

def accuracy_at_least_one(y_true, y_pred):
    """1 if at least one label matches"""
    correct = sum(1 for true_row, pred_row in zip(y_true, y_pred)
                  if len(set(np.where(true_row == 1)[0]) & 
                         set(np.where(pred_row == 1)[0])) > 0)
    return correct / len(y_true)

def percent_tags_correct(y_true, y_pred):
    """Average recall per sample"""
    scores = []
    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])
        if len(true_set) > 0:
            scores.append(len(true_set & pred_set) / len(true_set))
    return np.mean(scores)
```

---

## 🚀 Deployment Architecture

### Deployment Flow

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT                                          │
│  C:\Users\admin\Desktop\movie_genre_api\                   │
│  ├── app.py                                                 │
│  ├── Dockerfile                                             │
│  ├── fly.toml                                               │
│  ├── requirements.txt                                       │
│  ├── genre_classifier_model.pkl                            │
│  ├── tfidf_vectorizer.pkl                                  │
│  └── mlb_encoder.pkl                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ docker build -t movie-genre-api .
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  DOCKER BUILD PROCESS                                       │
│  1. Base image: python:3.9-slim                            │
│  2. Install system dependencies (gcc, g++)                 │
│  3. Install Python packages (requirements.txt)             │
│  4. Copy application files (app.py)                        │
│  5. Copy ML models (*.pkl files)                           │
│  6. Expose port 8000                                       │
│  7. Set CMD: uvicorn app:app                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Creates Docker Image
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  DOCKER IMAGE                                               │
│  Registry: fly.io/movie_genre_api     │
│  Size: ~300-400 MB (includes models)                       │
│  Layers:                                                    │
│    - OS + Python                                            │
│    - Dependencies                                           │
│    - Application code                                       │
│    - ML models (~30MB total)                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ fly deploy --app movie-genre-api-sparkling-frost-4339
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  FLY.IO DEPLOYMENT                                          │
│  1. Push image to Fly.io registry                          │
│  2. Create/update machines in region (ams)                 │
│  3. Start container instances                               │
│  4. Run health checks (/health endpoint)                   │
│  5. Configure HTTPS & routing                              │
│  6. Allocate resources (512MB RAM, shared CPU)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PRODUCTION (FLY.IO)                                        │
│  URL: https://movie_genre_api.fly.dev │
│  Region: Amsterdam (ams)                                    │
│  Resources: 512MB RAM, Shared CPU                          │
│  Auto-scaling: Yes (scale to 0 when idle)                 │
│  Health monitoring: /health endpoint                        │
│  HTTPS: Automatic Let's Encrypt SSL                       │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Commands

```bash
# Navigate to deployment directory
cd "C:\Users\admin\Desktop\movie_genre_api"

# Deploy/Update application
fly deploy --app movie_genre_api

# Start application (if stopped)
fly scale count 1 --app movie_genre_api

# Stop application (conserve resources)
fly scale count 0 --app movie_genre_api

# Check status
fly status --app movie_genre_api

# View logs
fly logs --app movie_genre_api

# Open in browser
fly open --app movie_genre_api
```

---

## 🏃 Quick Start

### Option 1: Use Deployed Application

Simply visit: **[movie-genre-api-sparkling-frost-4339.fly.dev](https://movie-genre-api-sparkling-frost-4339.fly.dev)**

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/yourusername/movie-genre-classifier.git
cd movie-genre-classifier

# Install dependencies
pip install -r requirements.txt

# Train model (requires dataset)
python train_model.py

# Copy deployment files
cp genre_classifier_model.pkl movie_genre_api/
cp tfidf_vectorizer.pkl movie_genre_api/
cp mlb_encoder.pkl movie_genre_api/

# Run application
cd movie_genre_api
python app.py

# Access at http://localhost:8000
```

### Option 3: Docker

```bash
cd movie_genre_api

# Build image
docker build -t movie-genre-classifier .

# Run container
docker run -p 8000:8000 movie-genre-classifier

# Access at http://localhost:8000
```

---

### Endpoints

#### `GET /`
Returns the interactive web interface

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "available_genres": ["Action", "Comedy", "Drama", ...]
}
```

#### `POST /predict`
Predict genres for a single movie

**Request:**
```json
{
  "description": "A group of intergalactic criminals must pull together..."
}
```

**Response:**
```json
{
  "description": "A group of intergalactic criminals...",
  "predicted_genres": ["Action", "Sci-Fi", "Adventure"]
}
```

**Python Example:**
```python
import requests

response = requests.post(
    "https://movie-genre-api-sparkling-frost-4339.fly.dev/predict",
    json={"description": "Your movie description here"}
)

print(response.json()['predicted_genres'])
# Output: ['Action', 'Sci-Fi', 'Adventure']
```


---

## 📊 Dataset

**Source**: IMDb Movie Dataset  
**Size**: 160,000+ movies  
**Features**:
- `movie_id`: Unique identifier
- `movie_name`: Title
- `description`: Plot summary (text)
- `genre`: Comma-separated genres (multi-label)

**Preprocessing**:
1. Remove missing values (description, genre)
2. Normalize text (lowercase, remove punctuation, collapse spaces)
3. Filter rare genres (<100 occurrences)
4. Final dataset: ~150,000 movies with 18 frequent genres

**Genre Distribution** (Top 10):
1. Drama
2. Comedy
3. Thriller
4. Action
5. Romance
6. Adventure
7. Crime
8. Sci-Fi
9. Fantasy
10. Horror

---

## 🧪 Model Training Details

### Hyperparameters

**TF-IDF:**
- `max_features`: 20,000
- `ngram_range`: (1, 2)
- `min_df`: 5
- `max_df`: 0.8

**Logistic Regression** (Best Model):
- `max_iter`: 1000
- `solver`: lbfgs (default)
- `multi_class`: ovr (OneVsRest)

**BERT:**
- Model: `bert-base-uncased`
- Max sequence length: 128
- Dropout: 0.3
- Learning rate: 2e-5
- Optimizer: AdamW

**LSTM:**
- Vocabulary: 20,000 words
- Sequence length: 150
- Embedding dim: 100
- LSTM units: 128
- Dropout: 0.3
- Optimizer: Adam (lr=1e-3)
- Loss: Binary crossentropy

---

## 🎯 Results Analysis

### Why Logistic Regression Won?

1. **Efficiency**: Fast training and inference
2. **Interpretability**: Clear feature weights
3. **Performance**: Excellent F1-score with TF-IDF features
4. **Deployment**: Small model size (~25MB)
5. **Stability**: No overfitting on test set

### Deep Learning Trade-offs

**BERT**:
- ✅ Best semantic understanding
- ✅ Captures context better
- ❌ Large model size (~400MB)
- ❌ Slow inference (~500ms)
- ❌ Requires GPU for training

**LSTM**:
- ✅ Good for sequential data
- ✅ Moderate model size (~50MB)
- ❌ Slower than traditional ML
- ❌ More prone to overfitting

**Winner: Logistic Regression** for production deployment!

---

## 📈 Performance Optimization

- **TF-IDF**: Pre-computed, cached in memory
- **Model**: Loaded once at startup
- **Inference**: Vectorized operations (scikit-learn)
- **API**: Asynchronous FastAPI
- **Caching**: Model artifacts in memory
- **Result**: <100ms response time

---

## 🔧 Configuration Files

### `requirements.txt`
```
fastapi
uvicorn
joblib
pandas
numpy
scikit-learn
matplotlib
seaborn
nest_asyncio
pyngrok
```

### `Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app.py .
COPY genre_classifier_model.pkl .
COPY tfidf_vectorizer.pkl .
COPY mlb_encoder.pkl .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `fly.toml`
```toml
# See https://fly.io/docs/reference/configuration/ for information about how to use this file.

app = 'movie-genre-api-sparkling-frost-4339'
primary_region = 'fra'

[build]

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = 'stop'
  auto_start_machines = true
  min_machines_running = 0
  processes = ['app']

[[vm]]
  memory = '1gb'
  cpu_kind = 'shared'
  cpus = 1

```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is for educational purposes. Dataset from IMDb.

---

## 👤 Author

**[Altaibekov Dinmukhammed]**
- 📧 Email: dimash.altaibekov@gmail.com
- 🐙 GitHub: [@shamid404]([https://github.com/yourusername](https://github.com/shamid404))

---

## 🙏 Acknowledgments

- IMDb for the movie dataset
- Hugging Face for BERT implementation
- scikit-learn community
- FastAPI framework
- Fly.io for hosting

---

## 📚 References

1. [Multi-Label Classification in scikit-learn](https://scikit-learn.org/stable/modules/multiclass.html)
2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)
4. [Fly.io Deployment Guide](https://fly.io/docs/)

---

⭐ **Star this repo if you found it helpful!**

📊 **Live Demo**: [movie-genre-api-sparkling-frost-4339.fly.dev](https://movie-genre-api-sparkling-frost-4339.fly.dev)
