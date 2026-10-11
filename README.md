# Customer Churn Analysis & Prediction

## Project Overview
This project analyzes customer behavior and builds machine-learning classification models to predict customer churn using an e-commerce customer dataset. It covers data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing, class-imbalance assessment, model comparison and tuning, evaluation, feature importance, and business recommendations.

## Objectives
- Understand customer behavior and the overall churn rate.
- Identify data-quality issues such as missing values and duplicate records.
- Explore numerical and categorical features associated with churn.
- Engineer useful behavioral features.
- Prepare numerical and categorical data for machine learning.
- Compare multiple classification algorithms.
- Tune a selected model and evaluate its performance.
- Identify important churn predictors and suggest retention strategies.

## Dataset
The project uses `ecommerce_customer_churn_dataset.csv`, containing 50,000 customer records.

**Target column:** `Churned`
- `0` = customer retained
- `1` = customer churned

Keep the CSV in the same folder as the notebook, or update the `DATA_PATH` variable in the notebook.

> **Note:** Confirm the business definition of churn before using these results operationally. The dataset may be synthetic or anonymized, so validate findings on representative production data before making business decisions.

## Project Files
```text
Customer-Churn-Analysis/
├── Customer_Churn_Analysis_and_Prediction.ipynb
├── ecommerce_customer_churn_dataset.csv
└── README.md
```

## Workflow
1. **Load and audit data** — inspect shape, data types, duplicates, missing values, and target balance.
2. **Clean data** — remove exact duplicates, validate target labels, normalize categorical text, and inspect extreme values.
3. **Explore data** — visualize churn distribution, feature distributions, correlations, and churn rates across customer segments.
4. **Engineer features** — derive behavioral indicators such as purchases per membership year, service calls per year, and inactivity flags when the source columns exist.
5. **Preprocess features** — impute missing numerical values, scale numerical features, and one-hot encode categorical features.
6. **Split data** — use a stratified train/test split to preserve the churn proportion.
7. **Address imbalance** — inspect class distribution and use class weighting for supported classifiers. Keep the test set at its natural distribution.
8. **Train and compare models** — compare Logistic Regression, Random Forest, Extra Trees, and HistGradient Boosting when available.
9. **Tune a model** — use stratified cross-validation and randomized hyperparameter search with ROC-AUC scoring.
10. **Evaluate** — report precision, recall, F1-score, ROC-AUC, classification report, confusion matrix, ROC curve, and precision-recall curve.
11. **Interpret and recommend** — rank predictors using permutation importance and translate meaningful signals into possible retention actions.

## Models and Evaluation Metrics
The notebook compares multiple classification algorithms and tunes a selected model.

- **Precision:** Among customers predicted to churn, the proportion who actually churned.
- **Recall:** Among customers who churned, the proportion correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks churners above retained customers across thresholds.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives.
- **Accuracy:** Overall proportion of correct predictions; do not use it alone when classes are imbalanced.

For churn prevention, recall can be important because missed churners may be lost customers. Precision is also important because retention campaigns have a cost. Choose the operating threshold according to business capacity and the relative costs of false positives and false negatives.

## Installation and Setup

### Requirements
- Python 3.10 or newer recommended
- Jupyter Notebook or Google Colab
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies in a terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run locally
1. Download the notebook and CSV dataset.
2. Place both files in the same directory.
3. Install the dependencies listed above.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Customer_Churn_Analysis_and_Prediction.ipynb`.
6. Run the cells from top to bottom.

### Run in Google Colab
1. Upload the notebook to Google Colab.
2. Upload `ecommerce_customer_churn_dataset.csv` to the Colab session.
3. Ensure `DATA_PATH` points to the uploaded CSV file.
4. Run the notebook cells from top to bottom.

## Results
The notebook produces:
- Data-quality summaries and descriptive statistics.
- EDA plots and churn-rate comparisons.
- A model-comparison table.
- Best cross-validation score and selected hyperparameters.
- Test-set classification metrics and confusion matrix.
- ROC and precision-recall curves.
- A feature-importance ranking.
- Practical retention recommendations based on observed predictors.

Model scores are generated when the notebook is executed in the target environment. Do not assume a model is production-ready based on a single test split; validate it on new data and monitor performance over time.

## Business Recommendations
Use the notebook's actual feature-importance results to decide which interventions to test. Potential actions include:
- Trigger win-back journeys when time since last purchase increases.
- Improve onboarding and engagement for customers whose login frequency declines.
- Reduce cart abandonment by simplifying checkout and clarifying delivery and return policies.
- Investigate repeated customer-service contacts and resolve recurring problems.
- Tailor loyalty benefits to customer behavior rather than offering blanket discounts.
- Use randomized holdout groups to measure incremental retention and retained margin.

These are hypotheses to test, not guaranteed causes of churn. Use controlled experiments to measure whether each action improves retention.

## Limitations
- Predictive importance indicates association, not causation.
- Results depend on dataset quality, representativeness, and the definition of churn.
- Thresholds should reflect outreach costs, available campaign capacity, and customer lifetime value.
- Monitor data drift, model performance, and segment-level differences after deployment.
- Validate the model on representative real-world data before using predictions for customer decisions.

## Author / Academic Use
This repository is suitable as an academic machine-learning project demonstrating an end-to-end churn analysis workflow. Add your name, course, institution, and submission date here if required.

## License
No license is specified. Add a license file if you intend to publish or redistribute the project.


