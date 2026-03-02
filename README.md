# Predicting Hotel Reviewer Score — Booking.com

## Overview

This notebook builds a machine learning pipeline to **predict reviewer scores** for hotels listed on Booking.com. It covers the full workflow: descriptive analysis, feature engineering, multicollinearity analysis, feature selection, and training a Random Forest regression model.

**Final model result: MAPE ≈ 0.161 (~16.1% mean absolute percentage error)**

---

## Dataset

- **Source:** Booking.com hotel reviews
- **File:** https://drive.google.com/drive/folders/1FsJ6OMdM_Ls_9PnameufpGWiYxgyf2-M?usp=sharing

### Columns

| Column | Description |
|---|---|
| `hotel_address` | Full hotel address |
| `hotel_name` | Hotel name |
| `average_score` | Overall hotel score on Booking.com |
| `reviewer_score` | Score given by the reviewer (**target variable**) |
| `reviewer_nationality` | Reviewer's country |
| `negative_review` | Text of negative review |
| `positive_review` | Text of positive review |
| `review_total_negative_word_counts` | Word count of negative review |
| `review_total_positive_word_counts` | Word count of positive review |
| `total_number_of_reviews` | Total reviews for the hotel |
| `total_number_of_reviews_reviewer_has_given` | Total reviews given by this reviewer |
| `additional_number_of_scoring` | Additional scoring count |
| `review_date` | Date of review |
| `days_since_review` | Days elapsed since the review |
| `tags` | List of tags describing the stay |
| `lat` / `lng` | Hotel geolocation (386,803 missing values) |

---

## Notebook Structure

### 1. Descriptive Data Analysis
- Loading `hotels.csv` and inspecting structure, data types, and shape.
- Identifying missing values: `lat` and `lng` columns have ~386k nulls.
- Removing duplicate rows.

### 2. Feature Engineering

Several new features are derived from the raw data:

**From `tags` (list of stay descriptors):**
- Boolean flags: `Leisure_trip`, `Couple`, `Stayed_1_night`, `Business_trip`, `Solo_traveler`
- `tag_count` — total number of tags per review

**From review text:**
- `negative_review` and `positive_review` encoded as binary (0 = no review text, 1 = has text)

**From `review_date`:**
- `review_date_m` — review month
- `review_date_qt` — review quarter

**From `days_since_review`:**
- Converted from string to numeric

**From `hotel_address`:**
- `city` — extracted manually (no geopy); results in 6 cities: London, Barcelona, Paris, Amsterdam, Vienna, Milan

### 3. Encoding & Scaling

- **Numerical features** scaled with `RobustScaler` (robust to outliers): `total_number_of_reviews`, `review_total_negative_word_counts`, `review_total_positive_word_counts`, `total_number_of_reviews_reviewer_has_given`, `additional_number_of_scoring`, `days_since_review`, `tag_count`
- **`city`** encoded with `OrdinalEncoder` (manual mapping)
- **`reviewer_nationality`** encoded with `BinaryEncoder`
- **`hotel_name`** encoded with `BinaryEncoder`
- Original columns (`hotel_address`, `review_date`, `hotel_name`, `reviewer_nationality`, `tags`, `lat`, `lng`) are dropped after encoding

### 4. Multicollinearity Analysis
- Correlation heatmap on numerical features.
- `additional_number_of_scoring` dropped due to high correlation with other features.

### 5. Feature Selection
- **Categorical features:** Chi-squared test (`chi2`) to rank feature importance vs. target.
- **Numerical features:** ANOVA F-test (`f_classif`) to rank feature importance vs. target.
- Low-importance encoded columns removed: `hotel_name_3–10`, `reviewer_nationality_0`, `reviewer_nationality_7`, `days_since_review`.

### 6. Modeling

- **Target variable:** `reviewer_score` (cast to integer)
- **Train/test split:** 75% / 25% (`random_state=42`)
- **Model:** `RandomForestRegressor` with 100 estimators
- **Evaluation metric:** MAPE (Mean Absolute Percentage Error)

```
MAPE: 0.1610
```

---

## Requirements

```
pandas
numpy
scikit-learn
category_encoders
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn category_encoders matplotlib seaborn
```

---

## How to Run

1. Place the dataset at `data/hotels.csv`.
2. Open the notebook:
   ```bash
   jupyter notebook rrf_hotels.ipynb
   ```
3. Run all cells top to bottom (`Kernel → Restart & Run All`).

---

## Environment

- **Python:** 3.9.10
- **Jupyter Notebook / JupyterLab**


