# Diabetic Retinopathy Detection System

A deep learning-based web application that analyzes retinal fundus images and predicts the severity of Diabetic Retinopathy (DR), built as a final year engineering project.

## About

Diabetic Retinopathy is a leading cause of vision loss in people with diabetes, and early detection significantly improves treatment outcomes. This project uses a ResNet50-based convolutional neural network trained on retinal imaging data to classify DR severity, wrapped in a simple Flask web app so users can upload an image and get an instant prediction.

**Built by a 4-member team** as part of our final year B.E. project (SMVITM, VTU).

**My contribution:** Front-end development of the Flask web interface (image upload flow and result visualization), preprocessing and organizing the training dataset, and overall task coordination across the team.

## Tech Stack

- Python
- TensorFlow / Keras
- ResNet50 (transfer learning)
- Flask
- HTML/CSS (frontend)

## How It Works

1. User uploads a retinal fundus image via the web interface (`index.html`)
2. The trained model (`train_model.py`) processes the image and predicts the DR severity stage
3. The result is displayed on `result.html`

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

## Dataset

Trained on a Kaggle retinal imaging dataset of 3,000+ labeled fundus images (`train.csv` contains the label mapping used for training).

## Result

Achieved ~85% classification accuracy on the validation set.
