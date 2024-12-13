# Mastering Sports Analytics: Predicting NBA Shot Success

This repository hosts the project files and code for a data science journey focused on predicting the likelihood of NBA shot success using data from 2003 to 2024. By applying advanced machine learning techniques and leveraging a robust dataset, we provide actionable insights into player performance and game strategies.

## Overview

Basketball analytics has revolutionized how teams strategize and evaluate performance. This project explores key factors influencing shot success, applies sophisticated models to predict outcomes, and highlights trends that can inform coaching and decision-making.

## Dataset and Scope

The project utilizes:
- **Shot Data**: Over 4 million records detailing shot types, distances, zones, and outcomes.
- **Player Data**: Comprehensive profiles, including positions and experience.
- **Team Data**: Information on all active NBA teams.

### Data Processing
- **Data Validation**: Fuzzy matching techniques (e.g., Qgram-Jaccard) ensured consistency between datasets.
- **Integration**: Linked shot, player, and team data to focus on active teams and players.

The refined dataset includes 1.3 million records, prepared for exploratory analysis and modeling.

## Exploratory Data Analysis (EDA)

Key insights from EDA:
- **Distance**: Shorter shots are more successful.
- **Action Types**: Dunks are most accurate, jump shots less so.
- **Game Context**: Home games see better accuracy; later quarters show a decline.

Interactive heatmaps and plots illustrate these trends. See the visualizations folder for details.

## Advanced Modeling Techniques

### Dimensionality Reduction
Principal Component Analysis (PCA) reduced the feature set from over 1,300 dimensions to 350, significantly optimizing model training without sacrificing variance.

### Machine Learning Models
We tested multiple models, including:
- **Tree-Based Algorithms**: Random Forests, Gradient Boosting.
- **Elastic Net Logistic Regression**: Selected for its balance of efficiency and accuracy (~66%).
- **Feedforward Neural Networks (FNNs)**: Achieved similar accuracy but required more resources.


### Feedforward Neural Network (FNN) Architecture
The FNN was designed with multiple layers, dropout, and ReLU activation functions to optimize prediction accuracy. The code is included in `fnn_model.py`.

## Model Selection and Evaluation

Grid search and randomized search were employed for hyperparameter tuning. Metrics used for evaluation include:
- **Confusion Matrices**
- **ROC and AUC Scores**

Best-performing models are saved in `best_models.pkl`.

