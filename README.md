# Sentiment Analysis 📊

A PyTorch-based sentiment analysis project designed for educational and research purposes. It features a two-layer LSTM model that classifies text as positive or negative with **just over 83%** validation accuracy. Additional CNN and transformer architectures are under development (currently achieving ~70% accuracy).

---

## 🚀 Features

- **Model Architecture:** Two-layer LSTM with 128 hidden units and 100-dimensional word embeddings (see `model.py`).
- **Regularization:** 50% dropout between LSTM layers to mitigate overfitting.
- **Configurable Training:** Adjust hyperparameters (learning rate, batch size, epochs) in `train.py`.
- **Inference:** Quick single-sentence prediction using `predict.py`.
- **Performance:** Consistent validation accuracy of ~83%.
- **Extensibility:** Easily integrate new models; CNN and transformer implementations are in progress.

---

## 🛠️ Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- numpy
- pandas
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Usage

### 1. Clone the repository

```bash
git clone https://github.com/yimango/Sentiment-analysis.git
cd Sentiment-analysis
```

### 2. Prepare your data

Place your CSV dataset files in a `data/` folder. Required columns:

| Column | Description                                 |
| ------ | ------------------------------------------- |
| text   | Input sentence or document                  |
| label  | Sentiment (0 = negative, 1 = positive)      |

### 3. Train the LSTM model

```bash
python train.py \
  --data_path data/train.csv \
  --epochs 10 \
  --batch_size 32 \
  --model_path sentiment_lstm_model.pth
```

### 4. Evaluate on validation data

```bash
python evaluate.py \
  --data_path data/val.csv \
  --model_path sentiment_lstm_model.pth
```

### 5. Run inference

```bash
python predict.py \
  --model_path sentiment_lstm_model.pth \
  --sentence "I love this movie!"
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "feat: add my feature"`)
4. Push the branch (`git push origin feature/my-feature`)
5. Open a pull request

---

## 💬 Support

For questions, bug reports, or feature requests, open an issue at [Sentiment-analysis/issues](https://github.com/yimango/Sentiment-analysis/issues).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute as needed.

