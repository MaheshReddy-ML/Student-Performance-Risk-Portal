from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


NUMERIC_FEATURES = [
    "age",
    "study_hours_per_day",
    "social_media_hours",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
]

CATEGORICAL_FEATURES = [
    "gender",
    "part_time_job",
    "diet_quality",
    "parental_education_level",
    "internet_quality",
    "extracurricular_participation",
]

FIELD_LABELS = {
    "age": "Age",
    "study_hours_per_day": "Study Routine",
    "social_media_hours": "Social Media",
    "attendance_percentage": "Attendance",
    "sleep_hours": "Sleep",
    "exercise_frequency": "Exercise",
    "mental_health_rating": "Mental Wellbeing",
    "gender": "Gender",
    "part_time_job": "Part-Time Work",
    "diet_quality": "Diet Quality",
    "parental_education_level": "Parental Education",
    "internet_quality": "Internet Quality",
    "extracurricular_participation": "Activities",
}

FIELD_OPTIONS = {
    "gender": ["Female", "Male", "Other"],
    "part_time_job": ["No", "Yes"],
    "diet_quality": ["Good", "Fair", "Poor"],
    "parental_education_level": ["Bachelor", "High School", "Master"],
    "internet_quality": ["Good", "Average", "Poor"],
    "extracurricular_participation": ["No", "Yes"],
}

FIELD_ALIASES = {
    "study_hours": "study_hours_per_day",
    "social_media_usage": "social_media_hours",
    "attendance": "attendance_percentage",
}

CHOICE_NORMALIZERS = {
    "gender": {
        "f": "Female",
        "female": "Female",
        "m": "Male",
        "male": "Male",
        "other": "Other",
    },
    "part_time_job": {
        "y": "Yes",
        "yes": "Yes",
        "n": "No",
        "no": "No",
    },
    "diet_quality": {
        "good": "Good",
        "fair": "Fair",
        "average": "Fair",
        "poor": "Poor",
    },
    "parental_education_level": {
        "high school": "High School",
        "highschool": "High School",
        "bachelor": "Bachelor",
        "master": "Master",
    },
    "internet_quality": {
        "good": "Good",
        "average": "Average",
        "poor": "Poor",
    },
    "extracurricular_participation": {
        "y": "Yes",
        "yes": "Yes",
        "n": "No",
        "no": "No",
    },
}


@dataclass
class PredictionResult:
    risk_level: str
    confidence: float
    probabilities: dict[str, float]
    summary: str
    advice: list[str]
    top_factors: list[dict[str, Any]]
    normalized_input: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(self.confidence, 4)
        return payload


class StudentRiskPredictor:
    def __init__(self, data_path: str | Path | None = None) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.data_path = Path(data_path) if data_path else self.base_dir / "student_habits.csv"
        self.dataset = self._load_dataset()
        self.feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.numeric_defaults = self._collect_numeric_defaults()
        self.categorical_defaults = self._collect_categorical_defaults()
        self.pipeline = self._build_pipeline()
        self.report = self._train()

    def _load_dataset(self) -> pd.DataFrame:
        frame = pd.read_csv(self.data_path)
        frame["risk_level"] = pd.cut(
            frame["exam_score"],
            bins=[-np.inf, 50, 70, np.inf],
            labels=["High", "Medium", "Low"],
            right=False,
        ).astype(str)
        frame = frame.drop(columns=["student_id", "exam_score", "netflix_hours"], errors="ignore")
        return frame

    def _collect_numeric_defaults(self) -> dict[str, float]:
        defaults = self.dataset[NUMERIC_FEATURES].median(numeric_only=True).to_dict()
        return {key: float(value) for key, value in defaults.items()}

    def _collect_categorical_defaults(self) -> dict[str, str]:
        defaults: dict[str, str] = {}
        for field in CATEGORICAL_FEATURES:
            mode = self.dataset[field].mode(dropna=True)
            defaults[field] = str(mode.iloc[0]) if not mode.empty else ""
        return defaults

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                    encoded_missing_value=-1,
                                ),
                            ),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ],
            verbose_feature_names_out=False,
        )

        classifier = RandomForestClassifier(
            n_estimators=320,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def _train(self) -> dict[str, Any]:
        features = self.dataset[self.feature_columns]
        target = self.dataset["risk_level"]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target,
        )

        self.pipeline.fit(x_train, y_train)

        predictions = self.pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]
        feature_names = list(preprocessor.get_feature_names_out())
        importances = classifier.feature_importances_

        self.feature_importances = {
            name: float(weight)
            for name, weight in sorted(
                zip(feature_names, importances),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        self.record_count = int(len(self.dataset))
        self.risk_distribution = {
            key: int(value)
            for key, value in self.dataset["risk_level"].value_counts().sort_index().to_dict().items()
        }
        self.model_accuracy = float(accuracy)
        self.training_report = report
        return {
            "accuracy": accuracy,
            "classification_report": report,
        }

    def overview(self) -> dict[str, Any]:
        top_features = [
            {
                "feature": FIELD_LABELS.get(name, name.replace("_", " ").title()),
                "importance": round(weight * 100, 1),
            }
            for name, weight in list(self.feature_importances.items())[:5]
        ]
        return {
            "record_count": self.record_count,
            "model_accuracy": round(self.model_accuracy * 100, 1),
            "risk_distribution": self.risk_distribution,
            "top_features": top_features,
            "field_options": FIELD_OPTIONS,
        }

    def normalize_input(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw_input)
        for source, target in FIELD_ALIASES.items():
            if source in payload and target not in payload:
                payload[target] = payload[source]

        normalized: dict[str, Any] = {}

        for field in NUMERIC_FEATURES:
            fallback = self.numeric_defaults[field]
            value = payload.get(field, fallback)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = fallback
            if field == "age":
                numeric_value = int(round(numeric_value))
                numeric_value = max(15, min(30, numeric_value))
            normalized[field] = numeric_value

        for field in CATEGORICAL_FEATURES:
            fallback = self.categorical_defaults[field]
            value = payload.get(field, fallback)
            text_value = str(value).strip()
            if not text_value:
                normalized[field] = fallback
                continue
            mapped = CHOICE_NORMALIZERS.get(field, {}).get(text_value.lower())
            normalized[field] = mapped or text_value.title()
            if field in FIELD_OPTIONS and normalized[field] not in FIELD_OPTIONS[field]:
                normalized[field] = fallback

        return normalized

    def _weight_factor(self, feature: str, base_weight: float) -> float:
        importance = self.feature_importances.get(feature, 0.0)
        return round(base_weight * (1.0 + (importance * 4.0)), 2)

    def _build_top_factors(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []

        def add_factor(feature: str, score: float, detail: str, recommendation: str) -> None:
            label = FIELD_LABELS.get(feature, feature.replace("_", " ").title())
            factors.append(
                {
                    "feature": label,
                    "direction": "risk" if score > 0 else "support",
                    "weight": round(abs(score), 2),
                    "detail": detail,
                    "recommendation": recommendation,
                }
            )

        attendance = payload["attendance_percentage"]
        if attendance < 75:
            add_factor(
                "attendance_percentage",
                self._weight_factor("attendance_percentage", 4.0),
                f"Attendance is {attendance:.1f}%, which is well below the strongest academic range.",
                "Build a weekly attendance target and protect the classes that carry the most marks.",
            )
        elif attendance < 85:
            add_factor(
                "attendance_percentage",
                self._weight_factor("attendance_percentage", 2.2),
                f"Attendance is {attendance:.1f}%, so consistency still needs some work.",
                "Push attendance closer to 90% to reduce avoidable academic pressure.",
            )
        else:
            add_factor(
                "attendance_percentage",
                -self._weight_factor("attendance_percentage", 1.3),
                f"Attendance at {attendance:.1f}% is acting as a protective signal.",
                "Keep the same attendance rhythm during busy assessment weeks.",
            )

        study_hours = payload["study_hours_per_day"]
        if study_hours < 2:
            add_factor(
                "study_hours_per_day",
                self._weight_factor("study_hours_per_day", 3.8),
                f"Study time is {study_hours:.1f} hours per day, which is low for steady exam preparation.",
                "Add one focused extra hour of revision on at least four days each week.",
            )
        elif study_hours < 3.5:
            add_factor(
                "study_hours_per_day",
                self._weight_factor("study_hours_per_day", 1.9),
                f"Study time is moderate at {study_hours:.1f} hours per day.",
                "Turn moderate study time into a stronger routine with a fixed daily schedule.",
            )
        else:
            add_factor(
                "study_hours_per_day",
                -self._weight_factor("study_hours_per_day", 1.4),
                f"{study_hours:.1f} study hours per day supports a more stable outlook.",
                "Keep your study sessions structured so the volume stays effective.",
            )

        sleep_hours = payload["sleep_hours"]
        if sleep_hours < 6:
            add_factor(
                "sleep_hours",
                self._weight_factor("sleep_hours", 3.0),
                f"Sleep is only {sleep_hours:.1f} hours, which can drag concentration down quickly.",
                "Move bedtime earlier and aim for at least 7 hours for the next two weeks.",
            )
        elif sleep_hours > 8.8:
            add_factor(
                "sleep_hours",
                self._weight_factor("sleep_hours", 1.2),
                f"Sleep is {sleep_hours:.1f} hours, which may be a sign of low energy or uneven routine.",
                "Check whether your sleep timing is regular and not cutting into study time.",
            )
        else:
            add_factor(
                "sleep_hours",
                -self._weight_factor("sleep_hours", 1.1),
                f"Sleep at {sleep_hours:.1f} hours is within a healthy range for learning.",
                "Protect that sleep window during exams and project deadlines.",
            )

        social_media_hours = payload["social_media_hours"]
        if social_media_hours > 4:
            add_factor(
                "social_media_hours",
                self._weight_factor("social_media_hours", 2.9),
                f"Social media time is {social_media_hours:.1f} hours, which is likely eating into revision.",
                "Trim social media usage by at least one hour and move that time into revision blocks.",
            )
        elif social_media_hours > 2.5:
            add_factor(
                "social_media_hours",
                self._weight_factor("social_media_hours", 1.5),
                f"Social media usage is noticeable at {social_media_hours:.1f} hours per day.",
                "Set an evening cutoff so scrolling does not spill into study or sleep time.",
            )
        else:
            add_factor(
                "social_media_hours",
                -self._weight_factor("social_media_hours", 0.9),
                f"Social media time is controlled at {social_media_hours:.1f} hours.",
                "Keep distractions limited while workload grows.",
            )

        mental_health_rating = payload["mental_health_rating"]
        if mental_health_rating <= 3:
            add_factor(
                "mental_health_rating",
                self._weight_factor("mental_health_rating", 2.7),
                f"Mental wellbeing is rated {mental_health_rating:.1f}/10, which signals real strain.",
                "Reduce pressure where possible and use campus, mentor, or family support this week.",
            )
        elif mental_health_rating <= 5:
            add_factor(
                "mental_health_rating",
                self._weight_factor("mental_health_rating", 1.6),
                f"Mental wellbeing is {mental_health_rating:.1f}/10 and could use support.",
                "Add recovery time between study sessions and review your workload realistically.",
            )
        else:
            add_factor(
                "mental_health_rating",
                -self._weight_factor("mental_health_rating", 1.0),
                f"Mental wellbeing at {mental_health_rating:.1f}/10 is helping performance stability.",
                "Keep the routines that are supporting your energy and focus.",
            )

        exercise_frequency = payload["exercise_frequency"]
        if exercise_frequency <= 1:
            add_factor(
                "exercise_frequency",
                self._weight_factor("exercise_frequency", 1.5),
                f"Exercise frequency is {exercise_frequency:.0f} days per week, which is quite low.",
                "Add light movement or walking on at least three days each week.",
            )
        elif exercise_frequency >= 4:
            add_factor(
                "exercise_frequency",
                -self._weight_factor("exercise_frequency", 0.8),
                f"Exercise on {exercise_frequency:.0f} days per week is supporting stamina.",
                "Keep your activity level steady during assessment periods.",
            )

        diet_quality = payload["diet_quality"]
        if diet_quality == "Poor":
            add_factor(
                "diet_quality",
                self._weight_factor("diet_quality", 1.5),
                "Diet quality is marked as poor, which can weaken focus and recovery.",
                "Aim for more consistent meals and hydration on study-heavy days.",
            )
        elif diet_quality == "Good":
            add_factor(
                "diet_quality",
                -self._weight_factor("diet_quality", 0.8),
                "Diet quality is marked as good and is adding stability to the routine.",
                "Keep meal timing consistent so energy stays level throughout the day.",
            )

        if payload["internet_quality"] == "Poor":
            add_factor(
                "internet_quality",
                self._weight_factor("internet_quality", 1.2),
                "Poor internet quality can disrupt online learning, resources, and planning.",
                "Download materials early and keep an offline backup for key subjects.",
            )

        if payload["part_time_job"] == "Yes" and study_hours < 3:
            add_factor(
                "part_time_job",
                self._weight_factor("part_time_job", 1.4),
                "A part-time job plus limited study hours may be stretching the schedule too thin.",
                "Protect non-negotiable revision slots before adding more work hours.",
            )

        if payload["extracurricular_participation"] == "Yes" and mental_health_rating >= 6:
            add_factor(
                "extracurricular_participation",
                -self._weight_factor("extracurricular_participation", 0.6),
                "Healthy extracurricular involvement may be supporting motivation and balance.",
                "Keep activities balanced so they refresh you without squeezing study time.",
            )

        ranked = sorted(factors, key=lambda item: item["weight"], reverse=True)
        return ranked[:5]

    def _build_summary(self, risk_level: str, confidence: float, top_factors: list[dict[str, Any]]) -> str:
        confidence_label = "high" if confidence >= 0.78 else "moderate" if confidence >= 0.6 else "cautious"
        lead = {
            "Low": "This profile looks stable overall and the model sees a lower academic risk pattern.",
            "Medium": "This profile sits in a watch zone, with a few habits that could pull performance down.",
            "High": "This profile shows multiple signals that usually line up with weaker academic performance.",
        }[risk_level]
        driver = top_factors[0]["feature"] if top_factors else "overall routine"
        return f"{lead} Confidence is {confidence_label} at {confidence * 100:.1f}%, and the strongest signal right now is {driver.lower()}."

    def _build_advice(self, risk_level: str, confidence: float, top_factors: list[dict[str, Any]]) -> list[str]:
        advice: list[str] = []

        opening = {
            "Low": "Keep reinforcing the routines that are already working instead of making drastic changes.",
            "Medium": "Focus on the top two weak areas first so improvements stay realistic and measurable.",
            "High": "Start with the most urgent habit changes this week rather than trying to fix everything at once.",
        }[risk_level]
        advice.append(opening)

        for factor in top_factors:
            recommendation = factor["recommendation"]
            if factor["direction"] == "risk" and recommendation not in advice:
                advice.append(recommendation)

        if confidence < 0.6:
            advice.append("Treat this prediction as a coaching signal, not a final judgment, because the model is less certain here.")

        if len(advice) == 1:
            advice.append("Your routine is already fairly balanced, so keep consistency high while exams approach.")

        return advice[:5]

    def predict(self, raw_input: dict[str, Any]) -> PredictionResult:
        normalized_input = self.normalize_input(raw_input)
        frame = pd.DataFrame([normalized_input], columns=self.feature_columns)
        probabilities_array = self.pipeline.predict_proba(frame)[0]
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        probabilities = {
            label: round(float(score) * 100, 1)
            for label, score in sorted(
                zip(classes, probabilities_array),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        risk_level = max(probabilities, key=probabilities.get)
        confidence = max(probabilities.values()) / 100
        top_factors = self._build_top_factors(normalized_input)
        summary = self._build_summary(risk_level, confidence, top_factors)
        advice = self._build_advice(risk_level, confidence, top_factors)
        return PredictionResult(
            risk_level=risk_level,
            confidence=confidence,
            probabilities=probabilities,
            summary=summary,
            advice=advice,
            top_factors=top_factors,
            normalized_input=normalized_input,
        )

    def predict_and_explain(self, raw_input: dict[str, Any]) -> tuple[str, float, str]:
        result = self.predict(raw_input)
        advice_lines = [result.summary, ""]
        advice_lines.extend(f"- {line}" for line in result.advice)
        return result.risk_level, result.confidence, "\n".join(advice_lines)


def prompt_for_cli_input() -> dict[str, Any]:
    return {
        "age": int(input("Age: ").strip() or 20),
        "gender": input("Gender (Female/Male/Other): ").strip() or "Female",
        "study_hours_per_day": float(input("Study hours per day: ").strip() or 3),
        "social_media_hours": float(input("Social media hours per day: ").strip() or 2),
        "attendance_percentage": float(input("Attendance percentage: ").strip() or 85),
        "sleep_hours": float(input("Sleep hours per day: ").strip() or 7),
        "diet_quality": input("Diet quality (Good/Fair/Poor): ").strip() or "Fair",
        "exercise_frequency": int(input("Exercise frequency per week (0-6): ").strip() or 3),
        "part_time_job": input("Part-time job (Yes/No): ").strip() or "No",
        "parental_education_level": input("Parental education (Bachelor/High School/Master): ").strip() or "Bachelor",
        "internet_quality": input("Internet quality (Good/Average/Poor): ").strip() or "Average",
        "mental_health_rating": int(input("Mental health rating (1-10): ").strip() or 6),
        "extracurricular_participation": input("Extracurricular participation (Yes/No): ").strip() or "Yes",
    }


PREDICTOR = StudentRiskPredictor()


def predict_and_explain(input_data: dict[str, Any]) -> tuple[str, float, str]:
    return PREDICTOR.predict_and_explain(input_data)


if __name__ == "__main__":
    outcome = PREDICTOR.predict(prompt_for_cli_input())
    print("\nRisk Level:", outcome.risk_level)
    print("Confidence:", f"{outcome.confidence * 100:.1f}%")
    print("Summary:", outcome.summary)
    print("Advice:")
    for line in outcome.advice:
        print("-", line)
