# PixelTruth
 
Detecting AI-generated images vs. real photographs using deep learning.
 
![PixelTruth UI](docs/screenshot.png)
 
## Overview
 
PixelTruth is a binary image classifier that distinguishes AI-generated images from real photographs. The project was built end-to-end — data collection, preprocessing, model training, a FastAPI backend, and a React frontend — as a hands-on exploration of the practical engineering challenges in AI-content detection, a real and growing problem as generative models become harder to distinguish from reality.
 
**Tech stack at a glance:** TensorFlow/Keras (EfficientNetB0 transfer learning), FastAPI, React, Google Cloud (Vertex AI, Cloud Run, Firebase Hosting — planned for full-scale training and deployment).
 
## Architecture
 
```
┌─────────────┐      ┌──────────────┐      ┌─────────────────────────┐
│   React     │─────▶│   FastAPI    │─────▶│  Model (.keras, local   │
│  Frontend   │◀─────│   Backend    │◀─────│  or Vertex AI Endpoint) │
└─────────────┘      └──────────────┘      └─────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Google Cloud     │
                    │  Storage (data,   │
                    │  model artifacts) │
                    └──────────────────┘
```
 
## Tech Stack
 
- **Model:** EfficientNetB0 (ImageNet-pretrained), two-stage transfer learning + fine-tuning, TensorFlow/Keras
- **Backend:** FastAPI (Python)
- **Frontend:** React
- **Cloud (planned, Part 2):** Vertex AI (full-scale training), Cloud Storage, Cloud Run, Firebase Hosting
- **Datasets:** CIFAKE (GAN-generated fakes), DiffusionDB (Stable Diffusion–generated fakes), COCO (real photographs)
## Project Structure
 
```
pixeltruth/
├── training/            # data fetching, preprocessing, training, evaluation scripts
├── backend/              # FastAPI app (app/main.py, model_loader.py, schemas.py)
├── frontend/              # React app
├── models/                # trained .keras files (gitignored)
├── data/                   # raw and processed datasets (gitignored)
├── docs/                   # screenshots, diagrams
└── infra/                   # cloudbuild.yaml (drafted, not yet run)
```
 
## Setup & Running Locally
 
**1. Clone and set up the training environment**
```bash
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
**2. Set up and run the backend**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export MODEL_SOURCE=local
export MODEL_PATH=../models/finetuned_v1.keras
uvicorn app.main:app --reload
```
 
**3. Set up and run the frontend**
```bash
cd frontend
npm install
npm run dev
```
 
The frontend calls the backend's `/predict` endpoint (`http://localhost:8000`), which loads the trained model and returns a label (`real` / `fake`) with a confidence score.
 
## How It Works
 
1. An image is uploaded through the React frontend.
2. The FastAPI backend resizes it to 224×224 and feeds it to an EfficientNetB0-based binary classifier.
3. The model outputs a sigmoid score; scores above 0.5 are classified `real`, below `fake` — mapped from the trained class ordering `['fake', 'real']`.
4. The label and confidence score are returned as JSON and displayed in the UI.
## Training Approach
 
The model went through an iterative improvement cycle after an initial baseline (trained solely on CIFAKE) revealed a **domain gap**: CIFAKE's "real" class is upscaled CIFAR-10 imagery (32×32 originals), not real photography, causing the baseline to misclassify genuine phone photos as AI-generated with high confidence.
 
**Fixes applied:**
1. **Real-photo replacement** — swapped CIFAKE's "real" class for actual photographs from COCO.
2. **Fake-class diversification** — combined CIFAKE (GAN-generated) with DiffusionDB (Stable Diffusion–generated), explicitly **balanced per generator family** rather than pooled-and-randomly-sampled, since early testing showed pooled sampling let CIFAKE's larger volume dominate and starve the model of diffusion examples.
3. **JPEG compression augmentation** — applied `tf.image.random_jpeg_quality` during training so the model is robust to the recompression real-world images undergo (social media, messaging apps).
4. **Two-stage fine-tuning** — trained the classification head with the EfficientNetB0 base frozen, then unfroze the top 30 layers and continued training at a low learning rate (1e-5), with BatchNorm layers kept in inference mode throughout (`training=False`) per standard transfer-learning practice.
**Results (held-out test set, 1,500 images):**
 
| Metric | Score |
|---|---|
| Overall accuracy | 98.27% |
| Fake precision / recall | 0.9751 / 0.9907 |
| Real precision / recall | 0.9905 / 0.9747 |
 
**Per-generator-family breakdown (fake class):**
 
| Source | Accuracy |
|---|---|
| CIFAKE (GAN) | 100.00% |
| DiffusionDB (diffusion) | 91.95% (up from 73.56% before source-balancing) |
 
## Known Limitations
 
- **Trained on two generator families.** The model has strong accuracy on GAN-based (CIFAKE) and diffusion-based (DiffusionDB/Stable Diffusion) fakes, but generalization to *other* generator families is unverified and likely weaker — this is a known, active limitation, not a bug. Testing against a Google Imagen-generated image (via Gemini) resulted in misclassification as "real," confirming the model has no learned basis for recognizing generators outside its training distribution.
- **No AI-detection model generalizes perfectly to unseen generators** — this is a documented open problem in this research area broadly, not unique to this project. New generators constantly shift what "fake" looks like.
- **Dataset scale is modest by design.** Training used roughly 10,000–15,000 total images (balanced across sources) to keep iteration fast on local hardware. Full-scale training (~70k+ images/class) via Vertex AI is planned for Part 2.
- **No adversarial robustness testing** has been performed — the model hasn't been evaluated against inputs deliberately crafted to fool it.
- **A third generator family (Midjourney) is currently being added** to the training mix locally, to test whether broader generator diversity improves generalization to unseen styles — results to be added once complete.
## Roadmap
 
- Add a third+ generator family (Midjourney, and ideally DALL·E/other diffusion variants) to training data
- Full-dataset training on Vertex AI (Part 2)
- Deploy backend to Cloud Run, frontend to Firebase Hosting
- Wire up `MODEL_SOURCE=vertex` in the backend to call a live Vertex AI Endpoint
- Model monitoring for prediction drift
- Frequency-domain (FFT/DCT) feature augmentation as a potential accuracy/robustness boost
## License
 
MIT License
 