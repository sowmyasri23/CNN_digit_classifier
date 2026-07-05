# 🔢 CNN Handwritten Digit Classifier

A Convolutional Neural Network built with **TensorFlow/Keras** to classify handwritten digits (0–9), trained on the scikit-learn `digits` dataset (1,797 8×8 grayscale images). Includes a live interactive **Streamlit** demo.

---

## 📊 Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **98.61%** (on 360 held-out images) |
| **Macro F1-score** | 0.986 |
| Weakest class | Digit 8 (91.4% recall) — most often confused with 1 and 9 |

**Confusion Matrix**

![Confusion Matrix](confusion_matrix.png)

**Training Curves**

![Training Curves](training_curves.png)

**Sample Predictions**

![Sample Predictions](sample_predictions.png)

---

## 🏗️ Architecture

```
Input (8x8x1)
  → Conv2D(32, 3x3, ReLU) → MaxPooling2D(2x2)
  → Conv2D(64, 3x3, ReLU) → Dropout(0.3)
  → Flatten → Dense(64, ReLU) → Dropout(0.3)
  → Dense(10, Softmax)
```

Trained for 30 epochs with the Adam optimizer and sparse categorical cross-entropy loss.

---

## 📁 Project Structure

```
├── train_model.py              # Data loading, model build/train/evaluate, saves visuals + model
├── app.py                      # Streamlit demo app
├── requirements.txt            # Dependencies
├── digit_cnn_model.h5          # Trained model weights
├── classification_report.txt   # Full per-class precision/recall/F1
├── confusion_matrix.png
├── training_curves.png
├── sample_predictions.png
└── README.md
```

---

## 🚀 Run it locally

```bash
git clone https://github.com/YOUR_USERNAME/cnn-digit-classifier.git
cd cnn-digit-classifier
pip install -r requirements.txt

python train_model.py     # retrain the model from scratch (optional — model is already included)
streamlit run app.py      # launch the interactive demo at localhost:8501
```

## ☁️ Run it in Google Colab

1. Open [Google Colab](https://colab.research.google.com) and upload `train_model.py`'s content (or the full notebook, if included).
2. Run all cells — training and evaluation plots render inline.
3. To try the live Streamlit demo from Colab, upload `app.py`, then run:
   ```python
   !pip install streamlit -q
   !streamlit run app.py &>/content/logs.txt &
   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
   !chmod +x cloudflared
   !./cloudflared tunnel --url http://localhost:8501
   ```
   Click the printed `trycloudflare.com` link to open the live app.

---

## 🎯 Demo

The Streamlit app lets you pick any test image via a slider and see:
- The digit image
- The model's predicted class
- Live confidence score
- Full class-probability bar chart

---

## 📝 Notes

- **Dataset choice:** scikit-learn's built-in `digits` dataset was used instead of full MNIST/Kaggle sets for guaranteed offline reproducibility — no external downloads required. It's lower resolution (8×8 vs. MNIST's 28×28), which is worth knowing if benchmarking against MNIST results.
- **Architecture simplicity:** kept intentionally shallow (2 conv layers) to match the small image size — deeper networks overfit quickly at this resolution.
- **Deployment:** demoed via Streamlit + a Cloudflare quick tunnel from Google Colab, since local Windows deployment hit environment-specific issues unrelated to the model code.

---

## 🔧 Tech Stack

`Python` · `TensorFlow` · `Keras` · `Scikit-learn` · `Streamlit` · `Matplotlib` · `NumPy`
