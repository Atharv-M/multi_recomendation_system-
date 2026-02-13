# Movie Recommendation System 🎬

A robust, hybrid movie recommendation engine built with FastAPI, utilizing Content-Based filtering, Collaborative Filtering (SVD), and Popularity-based models.

## 🚀 Features

-   **Hybrid Recommendation Engine**: Combines multiple strategies for better accuracy.
    -   **Collaborative Filtering**: Personalized recommendations using SVD (Singular Value Decomposition).
    -   **Content-Based**: Recommendations based on movie similarity (genres, features).
    -   **Popularity-Based**: Top-rated movies for new users (Cold Start problem).
-   **FastAPI Backend**: High-performance, async-ready API.
-   **Git LFS Integration**: Efficient handling of large machine learning models (>100MB).
-   **Clean Architecture**: Modular code structure separating data processing, modeling, and API routes.

## 🛠️ Tech Stack

-   **Python 3.10+**
-   **FastAPI** & **Uvicorn**
-   **Scikit-Learn** & **Surprise** (Recommendation algorithms)
-   **Pandas** & **Numpy** (Data manipulation)
-   **Git LFS** (Large File Storage)

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Atharv-M/multi_recomendation_system-.git
    cd multi_recomendation_system-
    ```

2.  **Pull Large Model Files (Important!)**
    This project uses Git LFS for model files. Ensure you have `git-lfs` installed.
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    # Create virtual environment
    python -m venv .venv
    
    # Activate virtual environment
    # Windows:
    # .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    
    # Install packages
    pip install -r requirements.txt
    ```

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:
-   **API Root**: [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/)
-   **Interactive Docs (Swagger UI)**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

## 📂 Project Structure

```
├── .gitattributes      # Git LFS configuration
├── .gitignore          # Git ignore rules
├── app/                # FastAPI application
│   ├── main.py         # App entry point
│   ├── routes/         # API endpoints
│   └── ...
├── artifacts/          # Trained models (tracked by LFS)
│   ├── collaborative/  # SVD models
│   ├── content/        # Content-based models
│   └── ...
├── data/               # Raw and processed data
├── src/                # Source code for models & processing
│   ├── models/         # Recommendation logic
│   └── ...
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
