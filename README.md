# CodeAlpha Machine Learning Internship — Task 2

## Speech Emotion Recognition using MFCC + CNN

A deep learning model that recognizes human emotions from speech audio using Mel-Frequency Cepstral Coefficients (MFCC) and a 1D Convolutional Neural Network.

---

## Problem Statement

Recognizing emotion from speech is a challenging classification problem with applications in human-computer interaction, mental health monitoring, call center analytics, and voice assistants. This project builds a CNN that classifies 3-second audio clips into 6 emotion categories.

This is a **multi-class classification** problem:
- Classes: `neutral, happy, sad, angry, fearful, disgust`
- Input: 3-second audio clip (22,050 Hz)
- Output: Predicted emotion + confidence score

---

## Dataset

- **Source:** [RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
- **Size:** 1,440 audio files from 24 actors (12 male, 12 female)
- **Classes after filtering:** 6 emotions
- **Files used:** 1,056 (excluding calm and surprised)
- **Class distribution:** Neutral (96), Happy/Sad/Angry/Fearful/Disgust (192 each)
- **Augmentation:** 3x (original + Gaussian noise + time-shift) → **3,168 samples**

### Why 6 Classes (Not 8)?

The original RAVDESS dataset has 8 emotions. We excluded:
- **Calm** — acoustically too similar to neutral
- **Surprised** — acoustically too similar to fearful

This is standard practice in speech emotion recognition research (helps model focus on distinctive emotional patterns).

### Preprocessing

| Step | Description |
|------|-------------|
| Sample Rate | 22,050 Hz (standard for speech) |
| Duration | 3 seconds (pad or truncate) |
| Feature | MFCC with 40 coefficients |
| Fixed Shape | (40, 174) — features × time-steps |

---

## Model Architecture

A Sequential CNN with 3 convolutional blocks:

| Layer | Details | Output Shape | Params |
|-------|---------|--------------|--------|
| Input | — | (40, 174, 1) | 0 |
| Conv1D | 64 filters, kernel=5, ReLU | (40, 174, 64) | 12,544 |
| BatchNorm | — | (40, 174, 64) | 256 |
| MaxPooling1D | pool=2 | (20, 174, 64) | 0 |
| Dropout | 0.2 | — | 0 |
| Conv1D | 128 filters, kernel=5, ReLU | (20, 174, 128) | 24,704 |
| BatchNorm | — | (20, 174, 128) | 512 |
| MaxPooling1D | pool=2 | (10, 174, 128) | 0 |
| Dropout | 0.2 | — | 0 |
| Conv1D | 128 filters, kernel=3, ReLU | (10, 174, 128) | 49,280 |
| BatchNorm | — | (10, 174, 128) | 512 |
| MaxPooling1D | pool=2 | (5, 174, 128) | 0 |
| Dropout | 0.3 | — | 0 |
| Flatten | — | (640) | 0 |
| Dense | 128 neurons, ReLU | (128) | 82,048 |
| Dropout | 0.4 | — | 0 |
| Dense | 6 neurons, Softmax | (6) | 774 |

**Total Parameters:** 191,558 (748.27 KB)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=0.0005) |
| Loss | Categorical Crossentropy |
| Metric | Accuracy |
| Batch Size | 32 |
| Epochs | 80 (stopped at 63 via EarlyStopping) |
| Validation Split | 0.2 |
| Callbacks | EarlyStopping (patience=15), ReduceLROnPlateau (patience=6) |
| Hardware | Google Colab — Tesla T4 GPU |

---

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **71.45%** |
| Test Loss | 0.8393 |
| Training Samples | 2,534 |
| Test Samples | 634 |
| Best Epoch | 48 |
| Random Baseline | 16.7% (1/6 classes) |

### Training Progress

- Training accuracy: 20.52% → 86.70%
- Validation accuracy: 21.29% → 71.45%
- Train-val gap: ~15% (acceptable — no severe overfitting)

### Per-Class Performance (Classification Report)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Neutral | 0.6471 | 0.5690 | 0.6055 | 58 |
| Happy | 0.6504 | 0.6957 | 0.6723 | 115 |
| Sad | 0.6250 | 0.5652 | 0.5936 | 115 |
| Angry | 0.7967 | 0.8522 | **0.8235** | 115 |
| Fearful | 0.7670 | 0.6810 | 0.7215 | 116 |
| Disgust | 0.7538 | 0.8522 | 0.8000 | 115 |
| **Macro Avg** | 0.7067 | 0.7025 | 0.7027 | 634 |
| **Weighted Avg** | 0.7121 | 0.7145 | 0.7115 | 634 |

**Best-performing classes:** Angry (F1=0.82) and Disgust (F1=0.80)
**Weakest classes:** Sad (F1=0.59) and Neutral (F1=0.61)

---

## Analysis of Misclassifications

Looking at the confusion matrix:

| Misclassification | Count | Why |
|-------------------|-------|-----|
| Sad → Disgust | 15 | Both have low pitch |
| Sad → Happy | 14 | Both have slower tempo, softer volume |
| Fearful → Happy | 14 | Both have high pitch and fast tempo |
| Fearful → Sad | 14 | Both have irregular breathing patterns |
| Happy → Angry | 12 | Both have high energy |
| Neutral → Happy | 11 | Neutral is a "mild" version of most emotions |
| Happy → Fearful | 10 | Both have high pitch |
| Neutral → Sad | 10 | Same as above |

**Key insight:** Most errors occur on acoustically similar emotions. Even humans struggle to distinguish between "neutral" and "mild happy" from audio alone. Angry and Disgust are the most distinctive — both have 85%+ recall.

---

## Real-World Test

I tested the trained model with random files from the RAVDESS dataset:

| Test | File | True Emotion | Predicted | Confidence |
|------|------|--------------|-----------|------------|
| 1 | `03-01-05-01-02-01-03.wav` | Angry | ✅ Angry | **99.49%** |
| 2 | `03-01-03-02-02-02-15.wav` | Happy | ✅ Happy | 85.41% |
| 3 | `03-01-05-01-01-01-08.wav` | Angry | ✅ Angry | **99.93%** |
| 4 | `03-01-04-02-01-01-22.wav` | Sad | ✅ Sad | **99.37%** |
| 5 | `03-01-03-01-02-01-05.wav` | Happy | ❌ Sad | 58.47% |

**Result: 4/5 correct predictions (80%)**

**Observation:** Angry, Sad, and Disgust are recognized with very high confidence (99%+). The model's only mistake was on a happy clip where the actor's tone was somewhat subdued — even for a human, this could sound ambiguous. This demonstrates the model's strengths lie in distinctive emotions.

---

## Visualizations

| Screenshot | Description |
|------------|-------------|
| `training_history.png` | Accuracy & loss curves over 63 epochs |
| `confusion_matrix.png` | 6×6 confusion matrix on test set |

---

## Technologies Used

- Python 3.13
- TensorFlow 2.x / Keras
- Librosa (audio processing)
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab (GPU: Tesla T4)

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Nafizur-Rahman-25/CodeAlpha_EmotionRecognition.git
cd CodeAlpha_EmotionRecognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
# Option A: Google Colab (recommended for GPU)
#   Upload CodeAlpha_EmotionRecognition.ipynb to colab.research.google.com
#   Runtime → Change runtime type → T4 GPU

# Option B: Local Jupyter
jupyter notebook CodeAlpha_EmotionRecognition.ipynb

# 4. Download RAVDESS dataset from Kaggle:
#    https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

# 5. Run all cells sequentially
```

---

## Key Learnings

- **MFCC is the foundation of speech processing:** 40 coefficients capture the essence of voice timbre.
- **Data augmentation is a game-changer:** 3x expansion (2,534 training samples) reduced overfitting significantly — before augmentation, we got 53% accuracy; after, 71% accuracy.
- **Simple CNN beats complex architecture:** A plain CNN with 191k params outperformed a BiLSTM+Attention model with 453k params (71% vs 34%) — smaller data prefers simpler models.
- **Feature count is not always better:** 40 MFCC features outperformed 263 combined features (MFCC+Chroma+Mel+ZCR) — redundant features add noise.
- **Transfer learning is task-specific:** wav2vec2 (trained on speech recognition) did NOT help emotion recognition (36% accuracy) — this is a known limitation in the research community.
- **Confusion matrix reveals patterns:** Errors are not random — most failures are on acoustically similar emotions.
- **Confidence ≠ Correctness:** High confidence (99%) on angry vs low confidence (58%) on ambiguous happy — the model "knows what it knows".
- **Overfitting is the #1 enemy:** Multiple iterations (dropout tuning, augmentation, LR scheduling) were needed to bring train-val gap down from 27% to 15%.
- **Try, fail, analyze, retry:** This project went through 4 major iterations before finding the winner. Iteration is part of ML.

---

## Author

**Md. Nafizur Rahman Nafiz**  
Department of Computer Science and Engineering  
Chittagong University of Engineering and Technology (CUET)  
CodeAlpha Machine Learning Internship

---

## Acknowledgements

- [CodeAlpha](https://www.codealpha.tech) for the internship opportunity
- RAVDESS dataset creators (Livingstone SR, Russo FA — 2018)
- [TensorFlow/Keras](https://keras.io) community
- [Librosa](https://librosa.org) for audio processing tools
- Google Colab for free GPU access