# EDiTH — AI-Powered Diabetic Retinopathy Screening & Clinical Decision-Support System

EDiTH is an AI-powered screening and clinical decision-support system built to bring Diabetic Retinopathy (DR) detection to rural and low-resource healthcare settings. It combines deep learning, explainable AI, and retrieval-augmented generation into a single pipeline — from raw fundus image to a clinician-readable report.

## Overview

Diabetic Retinopathy is a leading cause of preventable blindness, but timely screening is often inaccessible in rural areas due to a shortage of ophthalmologists and diagnostic infrastructure. EDiTH addresses this gap by automating fundus image analysis and generating interpretable AI-assisted reports.

## Key Features

- **Image Quality Assessment** — Automatically evaluates uploaded fundus images for blur, illumination, and artifacts before diagnosis, flagging unusable images for recapture.
- **Lesion Detection** — Uses image processing and deep learning to detect retinal lesions including microaneurysms, hemorrhages, and exudates.
- **DR Severity Classification** — Classifies each image into one of five DR severity levels (No DR, Mild, Moderate, Severe, Proliferative DR).
- **Grad-CAM Visual Explanations** — Generates class activation heatmaps to visually highlight the regions that influenced the model's prediction, supporting clinician trust and verification.
- **RAG-Based Case Retrieval** — Retrieves relevant, de-identified historical cases and validated clinical reports similar to the current finding, using a retrieval-augmented generation pipeline.
- **LLM-Generated Reports** — An LLM (via Groq API) synthesizes the DL model's output, clinical criteria, and retrieved cases into a clear, human-readable explanation and report — without altering or overriding the underlying model's diagnosis.

## System Architecture

```
Fundus Image
     │
     ▼
Image Quality Assessment ──► (Reject / Accept)
     │
     ▼
Lesion Detection & DR Classification (Deep Learning)
     │
     ├──► Grad-CAM Visual Evidence
     │
     ▼
RAG Retrieval (De-identified Historical Cases + Validated Reports)
     │
     ▼
LLM Report Generation (Groq API)
     │
     ▼
Clinician-Readable Report + Explanation
```

## Tech Stack

| Layer                | Technology                     |
|-----------------------|---------------------------------|
| Frontend              | HTML, CSS, JavaScript           |
| Backend               | FastAPI                         |
| Deep Learning          | Image processing + DL model for lesion detection and DR classification |
| Explainability         | Grad-CAM                        |
| Retrieval              | RAG (Retrieval-Augmented Generation) |
| LLM                     | Groq API                        |

## Project Structure

```
EDiTH/
├── frontend/               # HTML, CSS, JS client
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── models/          # DL model(s) for lesion detection & classification
│   │   ├── gradcam/         # Grad-CAM generation
│   │   ├── rag/              # Retrieval-augmented generation pipeline
│   │   ├── llm/               # Groq API integration & report generation
│   │   ├── routes/           # API endpoints
│   │   └── main.py
│   └── requirements.txt
├── data/                     # De-identified case data / knowledge base for RAG
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- A Groq API key

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

### Frontend Setup

The frontend is built with plain HTML, CSS, and JavaScript. Serve the `frontend/` directory using any static file server, or open `index.html` directly and point API calls to your running backend instance.

## How It Works

1. A fundus image is uploaded through the frontend.
2. The backend assesses image quality; poor-quality images are flagged for recapture.
3. The deep learning model detects lesions (microaneurysms, hemorrhages, exudates) and classifies DR severity into one of five levels.
4. Grad-CAM generates a heatmap highlighting the regions driving the prediction.
5. The RAG system retrieves de-identified historical cases and validated reports relevant to the current findings.
6. The LLM (via Groq API) combines the DL output, clinical criteria, and retrieved cases to generate a clear explanation and report — the model's diagnosis itself is never altered by the LLM.
7. The clinician receives the classification, Grad-CAM visualization, and generated report.

## Disclaimer

EDiTH is a decision-support and screening-assistance tool. It is intended to assist, not replace, qualified medical professionals. All outputs should be reviewed and confirmed by a certified ophthalmologist or healthcare provider before clinical action is taken.

## License

Add your license here (e.g., MIT, Apache 2.0).

## Acknowledgements

Built as part of an effort to improve DR screening accessibility in rural healthcare settings.
