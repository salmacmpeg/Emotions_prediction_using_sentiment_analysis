# Sentiment Analysis Project

This repository offers a rich exploration of **multiclass sentiment analysis and classification**, combining both traditional machine learning and deep learning approaches. It showcases how different embedding techniques and model architectures perform on nuanced emotional text data; from simple TF-IDF pipelines to transformer-powered classifiers.

This project experiments with NLP preprocessing, data loading, splitting, tokenization, vectorization, machine learning classifiers, deep learning classifiers with different architectures, fine-tuning a transformer model, all while addressing class imbalance, providing insightful evaluation approaches, and a final simple user interface.

## Project Highlights

* **Multiple Embedding Techniques**: TF-IDF vectorization and BERT tokenization.
* **Diverse Modeling Approaches**: XGBoost, Fully Connected Networks (FCN), RNNs, and Transformer-based models.
* **Class Imbalance Handling**: Sample weighting or inverse class weights.
* **Robust Evaluation**: Loss, accuracy, classification reports, confusion matrices, and performance diagrams.
* **Interactive Prediction**: Gradio-powered web interfaces for real-time sentiment classification.

## Project Structure

| File | Description |
|------|-------------|
| `TfidfVectorizer_Xgboost_sentiment_analysis.ipynb` | TF-IDF vectorization + XGBoost |
| `BertTokenizer_Xgboost_sentiment_analysis.ipynb` | BERT tokenization + XGBoost classifier |
| `FCN_Sentiment_analysis_pytorch.ipynb` | BERT tokenization + Fully Connected Network (PyTorch) |
| `RNN_sentiment_analysis_pytorch.ipynb` | BERT tokenization + Recurrent Neural Network (PyTorch) |
| `Transformer_sentiment_analysis_pytorch.ipynb` | BERT tokenization + BERT Transformer-based model + FCN layer (PyTorch) |
| `models/` | Directory for saving trained models and artifacts |

## Usage

1. **Install dependencies**  
   Ensure Python 3.x and required libraries are installed (see each script or notebook for details).

2. **Run a model**  
   Execute the notebook for step-by-step walkthroughs. Save and load models as you need.

3. **Try the Gradio interface**  
    A Gradio-powered UI for live sentiment prediction.

## Dataset

This project uses the **dair-ai/emotion** dataset from Hugging Face:

* **Classes**: sadness, joy, love, anger, fear, surprise
* **Format**: English text samples labeled with one of six emotion categories
* **Splits**: Training, validation, and test sets
* **Size**: ~20,000 samples

## Tokenization & Feature Extraction

* **BERT Tokenizer**:
  + Uses bert-base-uncased from Hugging Face Transformers
  + Outputs input_ids and attention_mask
  + Sequences padded/truncated to 90 tokens
  + Used for both PyTorch models and XGBoost (via token IDs or embeddings)
  ```python
    from transformers import AutoTokenizer
    model_name = "bert-base-uncased"
    autotokenizer = AutoTokenizer.from_pretrained(model_name)
    def _tokenize(batch):
        return autotokenizer(batch["text"], padding="max_length", truncation=True, max_length=embedding_dim)
    train_data_numericalized = train_data.map(_tokenize, batched=True, batch_size=len(train_data))
  ```
* **TF-IDF Vectorizer**:
  + Uses scikit-learn's TfidfVectorizer
  + Configured with:
    - max_features=10_000
    - ngram_range=(1,2)
    - min_df=1, max_df=0.9
  + Ideal for traditional ML models like XGBoost
  ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=.001,
    )
    tfidf = vectorizer.fit_transform(X_train)
  ```

---

## Modeling Approaches

* **BERT + XGBoost**:
  + Tokenize with BERT
  + Use input_ids or [CLS] embeddings as features
  + Train XGBoost with GPU acceleration and class weighting
* **TF-IDF + XGBoost**:
  + Vectorize text with TF-IDF
  + Train XGBoost with sample weights or SMOTE
* **FCN (PyTorch)**:
  + Simple fully connected layers
  + Uses BERT embeddings or token IDs
* **RNN (PyTorch)**:
  + Captures sequential dependencies
  + Uses embedding + GRU/LSTM layers
* **Transformer (PyTorch)**:
  + Custom transformer architecture
  + Trained end-to-end on emotion labels

## Saving & Loading Models

* **PyTorch models**: `torch.save()` and `torch.load()`
* **XGBoost models**: `model.save_model("xgb_model.json")` and `xgb.Booster.load_model()`
* **All models are stored in the models/ directory**

## Example Predictions

Here are some expanded test sentences used for evaluation:

```python
example_texts = [
    "iam happy to meet you",
    "i love you",
    "iam very upset iam sick",
    "i cant stand delaying my PhD defense any more",
    "i can punhs them in the face right now",
    "the hardest part about growing up is saying goodby to childhood dreams",
    "how did you do that to me?",
    "how dare you stand where he stod?"
]
for text in example_texts:
    sentiment, prob = predict_sentiment(text, my_model, autotokenizer, device, embedding_dim, classes)
    print(f"Text: {text}\nPredicted: {sentiment} ({prob:.3f})\n")
```

```python
# Gradio wrapper
import gradio as gr

# Gradio wrapper
def gradio_predict(text):
    sentiment, prob = predict_sentiment_one_example(text, xgb_model, vectorizer)
    return f"Predicted Emotion: {sentiment} ({prob:.3f} confidence)"

# Launch Gradio interface
gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter a sentence to analyze..."),
    outputs="text",
    title="Emotion Classifier",
    description="Enter a sentence and get its predicted emotion using XGBoost and TF-IDF/BERT features.",
    examples=[[text] for text in example_texts]

).launch()
```

## Use these to test your models interactively or in batch mode.

## Final Thoughts

This project is designed to be modular, extensible, and educational. Whether you're benchmarking traditional ML against transformers, experimenting with embeddings, or deploying a sentiment classifier, you'll find tools and examples to support your workflow.  
Feel free to contribute, extend, or adapt the models to your own datasets and tasks!

Made with ❤️ by Salma
