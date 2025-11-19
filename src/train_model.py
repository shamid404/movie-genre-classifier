import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/content/cleaned_imdb_by_movie_final.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

import pandas as pd
from collections import Counter
import re

def normalize_text(text):
    text = text.lower()                               # lowercase
    text = re.sub(r'[^a-z0-9\s]', ' ', text)          # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()          # collapse spaces
    return text

df_clean = df.dropna(subset=['description', 'genre']).copy()
print(f"Rows after removing NaN: {len(df_clean)}")
df_clean['description_norm'] = df_clean['description'].apply(normalize_text)
df_clean['genre_list'] = df_clean['genre'].apply(lambda x: [g.strip() for g in x.split(',')])
all_genres = [genre for genres in df_clean['genre_list'] for genre in genres]
genre_counts = Counter(all_genres)

print("\nTop 20 genres:")
for genre, count in genre_counts.most_common(20):
    print(f"{genre}: {count}")


# Visualize top genres
plt.figure(figsize=(12, 6))
top_genres = dict(genre_counts.most_common(15))
plt.bar(top_genres.keys(), top_genres.values())
plt.xticks(rotation=45, ha='right')
plt.title('Top 15 Movie Genres Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('genre_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Filter out very rare genres (optional - keep genres with at least 100 movies)
MIN_GENRE_COUNT = 100
frequent_genres = {genre for genre, count in genre_counts.items() if count >= MIN_GENRE_COUNT}
df_clean['genre_list'] = df_clean['genre_list'].apply(
    lambda genres: [g for g in genres if g in frequent_genres]
)
df_clean = df_clean[df_clean['genre_list'].apply(len) > 0]
print(f"\nRows after filtering rare genres: {len(df_clean)}")

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df_clean['genre_list'])
print(f"\nNumber of unique genres: {len(mlb.classes_)}")
print(f"Genre classes: {mlb.classes_}")

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)
X = tfidf.fit_transform(df_clean['description'])
print(f"\nFeature matrix shape: {X.shape}")

# Split data
# Split raw descriptions
X_text_train, X_text_test, y_train_dl, y_test_dl = train_test_split(
    df_clean['description'], y, test_size=0.2, random_state=42
)

# ====================== BERT TOKENIZER AND DATASETS ======================
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert text Series to plain list to avoid pandas indexing issues
train_dataset = MovieDataset(list(X_text_train), np.array(y_train_dl), tokenizer)
test_dataset = MovieDataset(list(X_text_test), np.array(y_test_dl), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# ====================== CUSTOM METRICS ======================

def partial_accuracy(y_true, y_pred):
    """Jaccard-like partial accuracy: intersection / union"""
    scores = []

    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])

        if len(true_set) == 0 and len(pred_set) == 0:
            scores.append(1)
            continue

        union = len(true_set | pred_set)
        if union == 0:
            scores.append(0)
            continue

        score = len(true_set & pred_set) / union
        scores.append(score)

    return np.mean(scores)


def accuracy_at_least_one(y_true, y_pred):
    """1 if at least one label matches, else 0"""
    correct = 0
    total = len(y_true)

    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])

        if len(true_set & pred_set) > 0:
            correct += 1

    return correct / total


def percent_tags_correct(y_true, y_pred):
    """How many true labels were guessed correctly (per sample)"""
    scores = []

    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])

        if len(true_set) == 0:
            continue

        score = len(true_set & pred_set) / len(true_set)
        scores.append(score)

    return np.mean(scores)



# ====================== DEEP LEARNING MODELS ======================
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------- BERT ----------------------
class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return torch.sigmoid(self.out(output))  # multi-label

# Prepare BERT tokenizer and datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Convert text Series to plain list to avoid pandas indexing issues
train_dataset = MovieDataset(list(X_text_train), np.array(y_train_dl), tokenizer)
test_dataset = MovieDataset(list(X_text_test), np.array(y_test_dl), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertClassifier(n_classes=y.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = AdamW(bert_model.parameters(), lr=2e-5)

# Simple training loop (1 epoch for example)
for epoch in range(1):
    bert_model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("BERT training done!")
# Compute custom metrics for BERT
print("BERT Metrics:")
print("Partial Accuracy:", partial_accuracy(y_test_dl, bert_preds_bin))
print("Accuracy ≥1:", accuracy_at_least_one(y_test_dl, bert_preds_bin))
print("% True Labels Guessed:", percent_tags_correct(y_test_dl, bert_preds_bin))


# ---------------------- LSTM ----------------------
MAX_WORDS = 20000
MAX_LEN = 150
EMBED_DIM = 100

tokenizer_keras = Tokenizer(num_words=MAX_WORDS)
tokenizer_keras.fit_on_texts(df_clean['description'])

X_seq = tokenizer_keras.texts_to_sequences(df_clean['description'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN)

X_train_pad, X_test_pad = train_test_split(X_pad, test_size=0.2, random_state=42)
y_train_dl, y_test_dl = train_test_split(y, test_size=0.2, random_state=42)

lstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(y.shape[1], activation='sigmoid')  # multi-label
])

lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train_dl, batch_size=32, epochs=3, validation_split=0.1)

print("LSTM training done!")

# ---------------------- PREDICTIONS ----------------------
# BERT predictions
bert_model.eval()
bert_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = bert_model(input_ids, attention_mask)
        bert_preds.extend(outputs.cpu().numpy())
bert_preds = np.array(bert_preds)
bert_preds_bin = (bert_preds > 0.5).astype(int)

# LSTM predictions
lstm_preds = lstm_model.predict(X_test_pad)
lstm_preds_bin = (lstm_preds > 0.5).astype(int)

# Compute custom metrics for BERT
print("BERT Metrics:")
print("Partial Accuracy:", partial_accuracy(y_test_dl, bert_preds_bin))
print("Accuracy ≥1:", accuracy_at_least_one(y_test_dl, bert_preds_bin))
print("% True Labels Guessed:", percent_tags_correct(y_test_dl, bert_preds_bin))

# Compute custom metrics for LSTM
print("LSTM Metrics:")
print("Partial Accuracy:", partial_accuracy(y_test_dl, lstm_preds_bin))
print("Accuracy ≥1:", accuracy_at_least_one(y_test_dl, lstm_preds_bin))
print("% True Labels Guessed:", percent_tags_correct(y_test_dl, lstm_preds_bin))

# ====================== MODELS ======================

models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
    'Naive Bayes': OneVsRestClassifier(MultinomialNB()),
    'Linear SVM': OneVsRestClassifier(LinearSVC(max_iter=1000, random_state=42))
}

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print('='*60)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    hamming = hamming_loss(y_test, y_pred)

    # Custom metrics
    p_acc = partial_accuracy(y_test, y_pred)
    one_acc = accuracy_at_least_one(y_test, y_pred)
    percent_correct = percent_tags_correct(y_test, y_pred)

    # Save results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hamming_loss': hamming,
        'partial_accuracy': p_acc,
        'accuracy_at_least_one': one_acc,
        'percent_tags_correct': percent_correct,
        'model': model
    }

    # Print results
    print(f"\nResults for {name}:")
    print(f"Accuracy:                 {accuracy:.4f}")
    print(f"Precision (weighted):     {precision:.4f}")
    print(f"Recall (weighted):        {recall:.4f}")
    print(f"F1-Score (weighted):      {f1:.4f}")
    print(f"Hamming Loss:             {hamming:.4f}")

    print("\n--- Custom Multilabel Metrics ---")
    print(f"Partial Accuracy:         {p_acc:.4f}")
    print(f"Accuracy (≥1 label):      {one_acc:.4f}")
    print(f"% of True Labels Guessed: {percent_correct:.4f}")


# ====================== SELECT BEST MODEL ======================

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\n{'='*60}")
print(f"Best model by F1-score: {best_model_name}")
print('='*60)

y_pred_best = best_model.predict(X_test)





# Classification report per genre
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=mlb.classes_, zero_division=0))



# Confusion matrix for each genre
conf_matrices = multilabel_confusion_matrix(y_test, y_pred_best)
num_genres = len(mlb.classes_)
rows = int(np.ceil(num_genres / 3))  # 3 matrices per row

fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
axes = axes.ravel()

for idx in range(num_genres):
    sns.heatmap(conf_matrices[idx], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(mlb.classes_[idx])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

# Remove unused axes if any
for j in range(num_genres, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model
joblib.dump(best_model, 'genre_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(mlb, 'mlb_encoder.pkl')

print("\nModel and artifacts saved successfully!")
print("- genre_classifier_model.pkl")
print("- tfidf_vectorizer.pkl")
print("- mlb_encoder.pkl")

def predict_genres(description, model, vectorizer, encoder):
    """Predict genres for a movie description"""
    X_new = vectorizer.transform([description])
    y_pred = model.predict(X_new)
    predicted_genres = encoder.inverse_transform(y_pred)
    return predicted_genres[0] if predicted_genres[0] else ['Unknown']

# Test with sample descriptions
test_descriptions = [
    "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe.",
    "A young wizard begins his magical education at a school of witchcraft and wizardry.",
    "A computer hacker learns about the true nature of his reality and his role in the war against its controllers."
]

print("\n" + "="*50)
print("Sample Predictions:")
print("="*50)
for desc in test_descriptions:
    predicted = predict_genres(desc, best_model, tfidf, mlb)
    print(f"\nDescription: {desc[:80]}...")
    print(f"Predicted Genres: {', '.join(predicted)}")


# ===========================
# 9. MODEL COMPARISON VISUALIZATION
# ===========================

# Create comparison dataframe (standard metrics)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()]
})

# Create custom metrics dataframe
custom_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Partial Accuracy': [results[m]['partial_accuracy'] for m in results.keys()],
    'Accuracy ≥1 Match': [results[m]['accuracy_at_least_one'] for m in results.keys()],
    '% True Labels Guessed': [results[m]['percent_tags_correct'] for m in results.keys()]
})


# ===========================
# PLOT 1 – Standard Metrics
# ===========================

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.2

ax.bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy')
ax.bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision')
ax.bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall')
ax.bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (Standard Metrics)')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_standard.png', dpi=300, bbox_inches='tight')
plt.show()


# ===========================
# PLOT 2 – Custom Multilabel Metrics
# ===========================

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(custom_df))
width = 0.25

ax.bar(x - width, custom_df['Partial Accuracy'], width, label='Partial Accuracy')
ax.bar(x, custom_df['Accuracy ≥1 Match'], width, label='Accuracy ≥1 Label')
ax.bar(x + width, custom_df['% True Labels Guessed'], width, label='% True Labels Guessed')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (Custom Multilabel Metrics)')
ax.set_xticks(x)
ax.set_xticklabels(custom_df['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_custom.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nTraining completed! Model ready for deployment.")
print(f"Best model ({best_model_name}) saved and ready to use in FastAPI/Flask app.")
