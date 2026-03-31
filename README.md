## 🌐 Live Deployment

- API: https://student-performance-risk-portal-2.onrender.com  

# Student Performance Risk Portal

A production-style student performance risk prediction project built with Python, scikit-learn, and a single-page frontend.

This application analyzes student habit and lifestyle inputs, predicts an academic risk level, and returns a confidence score with practical advice. The project includes:

- A trained machine learning pipeline
- A local Python backend server
- A polished single-page frontend
- A dataset-driven risk overview
- A clean project structure for local development and GitHub publishing

## Project Overview

The goal of this project is to estimate a student's academic risk profile based on behavior and support-related inputs such as:

- Study hours
- Attendance percentage
- Sleep hours
- Social media usage
- Mental health rating
- Exercise frequency
- Diet quality
- Internet quality
- Part-time work
- Extracurricular participation

The system classifies each profile into one of three categories:

- `Low`
- `Medium`
- `High`

It also returns:

- Prediction confidence
- Probability distribution across all risk classes
- Top influencing signals
- Actionable improvement suggestions

## Features

- Modern single-page UI with a production-ready dashboard layout
- Local backend with JSON API endpoints
- Machine learning pipeline trained from `student_habits.csv`
- Input normalization and safe default handling
- Risk insights, probability bars, and action recommendations
- File guide section built directly into the frontend
- No frontend build step required

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- HTML5
- CSS3
- Vanilla JavaScript
- Python standard library HTTP server

## Project Structure

```text
student/
├── app.py
├── index.html
├── model.py
├── requirements.txt
├── student_habits.csv
├── .gitignore
└── README.md
```

## GitHub Repository Setup

Your GitHub profile name is:

```text
MaheshReddy-ML
```

If you publish this project with a repository name like `student-performance-risk-portal`, the clone URL would look like this:

```bash
git clone https://github.com/MaheshReddy-ML/student-performance-risk-portal.git
```

If you choose a different repository name, replace the last part of the URL accordingly:

```bash
git clone https://github.com/MaheshReddy-ML/<student-performance-risk-portal>.git
```

## How to Clone the Project

```bash
git clone https://github.com/MaheshReddy-ML/<student-performance-risk-portal>.git
cd <student-performance-risk-portal>/student
```

If the `student` folder itself is the root of your GitHub repository, then use:

```bash
git clone https://github.com/MaheshReddy-ML/<student-performance-risk-portal>.git
cd <student-performance-risk-portal>
```

## Requirements

Before running the project, make sure you have:

- Python 3.10 or later
- `pip`

The project dependencies are listed in `requirements.txt`.

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

From inside the `student` project directory:

```bash
python3 app.py
```

By default, the server starts at:

```text
http://127.0.0.1:8000
```

Open that URL in your browser to use the application.

### Run on a custom host and port

```bash
python3 app.py --host 0.0.0.0 --port 8000
```

## How the Application Works

### Frontend

The frontend is stored in `index.html` and includes:

- The full UI
- Styling
- Form handling
- API calls to the backend
- Dynamic result rendering

### Backend

The backend is in `app.py` and:

- Serves the frontend
- Exposes API routes
- Loads the trained predictor from `model.py`
- Returns health and prediction data as JSON

### Model Layer

The machine learning logic is in `model.py` and:

- Loads the dataset
- Creates the target risk labels
- Builds the preprocessing and classification pipeline
- Trains the model
- Generates predictions and explanation data

## API Endpoints

### `GET /api/health`

Returns:

- Application status
- Model accuracy
- Record count
- Risk distribution
- Top features
- Project file metadata

Example:

```bash
curl http://127.0.0.1:8000/api/health
```

### `POST /api/predict`

Accepts a JSON payload and returns:

- `risk_level`
- `confidence`
- `probabilities`
- `summary`
- `advice`
- `top_factors`
- `normalized_input`

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 19,
    "gender": "Male",
    "study_hours_per_day": 1.4,
    "attendance_percentage": 69,
    "sleep_hours": 5.3,
    "social_media_hours": 5.0,
    "mental_health_rating": 3,
    "exercise_frequency": 1,
    "diet_quality": "Poor",
    "internet_quality": "Poor",
    "part_time_job": "Yes",
    "extracurricular_participation": "No",
    "parental_education_level": "High School"
  }'
```

## Example Workflow

1. Start the server with `python3 app.py`
2. Open `http://127.0.0.1:8000`
3. Fill in the student assessment form
4. Submit the form
5. Review the predicted risk level, confidence, and suggested actions

## Notes

- The model trains when the application starts.
- No React, Node.js, or frontend build tooling is required.
- The frontend and backend are designed to work locally out of the box.
- The UI is mobile-friendly and also works well on desktop.

## Future Improvements

- Save prediction history
- Add authentication for multiple users
- Export prediction reports as PDF
- Add charts for trend tracking
- Deploy the app to a cloud platform

## Author

Mahesh Reddy  
GitHub: [MaheshReddy-ML](https://github.com/MaheshReddy-ML)
