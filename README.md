# LingoStress: AI-Powered Pronunciation Assessment
---

**LingoStress** is an interactive web application designed to help English
learners master their pronunciation. Unlike standard speech recognition tools,
LingoStress focuses specifically on **Syllable Stress** (rhythm) and **Phoneme
Accuracy** (articulation), providing granular, visual feedback on exactly which
parts of a word were mispronounced.

## 🚀 Key Features

  * **Real-time Audio Analysis:** Record your voice directly in the browser.
  * **Syllable Stress Detection:** AI models analyze your pitch and energy to
  determine if you emphasized the correct syllable (e.g., *re-CORD* vs.
  *RE-cord*).
  * **Phoneme-Level Scoring:** Uses advanced Goodness of Pronunciation (GOP)
  algorithms to score individual sounds (e.g., checking if you pronounced the
  "th" in "think" correctly).
  * **Visual Feedback:** Color-coded heatmaps and stress bars make it easy to
  understand errors.
  * **Difficulty Bands:** Practice words categorized by difficulty levels.

## 🛠 Tech Stack

**Backend (API):**

  * **Python 3.11.9+**
  * **FastAPI:** High-performance web framework.
  * **TensorFlow/Keras:** For the LSTM-based Syllable Stress detection model.
  * **PyTorch & HuggingFace Transformers:** For the Wav2Vec2-based Phoneme
  recognition (GOP).
  * **Librosa:** For advanced audio signal processing (silence trimming,
  normalization).

**Frontend (UI):**

  * **React (TypeScript):** Component-based UI architecture.
  * **Vite:** Fast build tool.
  * **Tailwind CSS:** For modern, responsive styling.
  * **Lucide React:** For UI icons.

-----

## 📂 Project Structure

Basic components of the project are organized as follow:

```text
project_root/
├── server/                     # Python Backend Code
│   ├── main.py                 # API Entry point
│   ├── requirements.txt        # Python dependencies
│   ├── setup.py                # Python package setup
│   ├── services/               # AI Logic (Stress & GOP services)
│   └── models/                 # Model checkpoints
│       ├── sylstress           # LSTM Model for stress detection
│       └── ctcgop              # Fine-tuned Wav2Vec2 Model (Phoneme level)
│   └── utils/                  # Helper functions (Audio extraction)
├── sylstress/                  # Training code for stress model
└── frontend/                   # React Application (Source code)
```

-----

## ⚡ Installation Guide

### Prerequisites

  * **Python 3.11.9** is recommended.
  * **Node.js 18** or higher.
  * **Git LFS** (Large File Storage) is required to download the acoustic models.

### 1\. Backend Setup (Python)

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Somethings1/pronunciation-assessment.git
    cd pronunciation-assessment
    ```

2.  **Set up a Virtual Environment:**

    ```bash
    # Create venv
    python -m venv venv

    # Activate venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    First, install the project in editable mode (this links the `app` folder):

    ```bash
    cd server
    pip install -e .
    ```

    Then install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```


### 2\. Frontend Setup (React)

1.  **Navigate to the frontend folder:**

    ```bash
    cd ../frontend
    ```

2.  **Install Node packages:**

    ```bash
    npm install
    ```

3.  **Configure Environment Variables:**
    Create a `.env.local` file in the `frontend` folder:

    ```properties
    VITE_ASSESSMENT_API_URL=http://127.0.0.1:8000
    ```

-----

## 🖥 Usage Guide

### 1\. Start the Backend Server

Open a terminal in the **root** folder (ensure `venv` is active):

```bash
uvicorn server.main:app --reload
```

*The API will start at `http://127.0.0.1:8000`. You can verify it's running by visiting `http://127.0.0.1:8000/docs`.*

### 2\. Start the Frontend Application

Open a new terminal in the **frontend** folder:

```bash
cd frontend
npm run dev
```

*The application will typically start at `http://localhost:5173`.*

### 3\. How to Practice

1.  Open the web app in your browser.
2.  Select a **Difficulty Band**.
3.  Click the **Microphone** icon and read the displayed word aloud.
4.  Wait for the analysis.
5.  **Review your results:**
      * **Green Blocks:** Good pronunciation.
      * **Yellow/Red Blocks:** Needs improvement (wrong sound or wrong stress).
      * **Stress Bars:** Compare the "Target" rhythm vs. "You" to see if you emphasized the correct syllable.

### 4\. Train stress model

The stress model is pretrained on L2-ARCTIC dataset. However, if you wish to
train it on your own dataset, explore the sylstress folder at top level.
