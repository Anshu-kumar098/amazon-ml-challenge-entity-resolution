# Amazon ML Challenge - Entity Resolution Pipeline
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Polars](https://img.shields.io/badge/Polars-Engineered-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

##  Project Overview
This repository contains my machine learning pipeline developed for the Amazon ML Challenge. The core objective was to perform automated **entity resolution (record linkage)** to match, clean, and unify product records efficiently at scale using advanced string similarity and gradient boosting.

##  Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Polars
* **String Similarity & Matching:** RapidFuzz
* **Machine Learning:** XGBoost, Scikit-Learn
* **Environment:** Google Colab
## Text-based flowchart
[Raw TSV Data (Google Drive)]
>>
1_preprocessing.ipynb  ──► (Cleaned Dataframes)
>>
2_pair_creation.ipynb  ──► (Candidate Pairs & Feature Engineering)
>>
3_model_training.ipynb ──► (XGBoost Classification & Final Output)

##  Approach & Methodology
1. **Data Preprocessing & Cleansing:** Handled missing values, standardized text formats, and optimized data types using **Polars** for high-speed dataframe operations.
2. **Feature Engineering:** 
   * Computed string distance metrics and token ratios using **RapidFuzz**.
   * Engineered **custom structural features**, including string length comparisons and first-character acronyms (extracted from sorted words) to capture formatting patterns and abbreviations between record pairs.
3. **Classification Model:** Trained an **XGBoost** classifier on the engineered feature set to accurately predict matching entity pairs.
4. **Inference & Submission:** Generated predictions and formatted the final output (`matching_results.tsv`) according to competition criteria.

## ⚙️ How to Run
1. Clone the repository:
  ```bash
git clone https://github.com/Anshu-kumar098/amazon-ml-challenge-entity-resolution.git
cd amazon-ml-challenge-entity-resolution
