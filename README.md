# H\&M Personalized Fashion Recommendations

👉 **[Click here to view the complete Project Case Study with all code and visual plots](H%26M%20Recommendations.md)**

#### **Project Overview**



This project tackles the Kaggle H\&M Personalized Fashion Recommendations challenge. The objective is to predict what articles each of the 1.3 million customers will purchase in the 7-day period immediately following the training data.



**Final Performance:** Achieved a MAP@12 score of 0.02192, significantly outperforming the "top-12 popular" baseline.



This project focuses on building a highly scalable, memory-efficient **Two-Stage Retrieval** and **Ranking Architecture**, optimizing for top-line revenue generation through targeted personalization.



#### **System Architecture**



To handle the immense scale of the dataset (millions of transactions and customers) without exceeding RAM limits, the recommendation pipeline is split into two distinct stages:



###### **Stage 1: Multi-Source Retrieval (Candidate Generation)**



Rather than scoring the entire 100,000+ item catalog for every user, Stage 1 uses fast heuristics to generate a high-probability pool of ~50 candidate items per customer. Candidates are sourced from:



* **Collaborative Filtering / Association:** Items frequently bought together with a user's recent purchases.



* **Demographic Trends:** Top-selling items tailored to the user's specific age group.



* **Seasonal Popularity:** Items that historically spike in sales during the target month.



* **Purchase History:** Items the user has bought previously (accounting for fast-fashion replenishment).



###### **Stage 2: LightGBM Ranker (Precision Scoring)**



A Gradient Boosted Decision Tree (LGBMRanker) evaluates the ~50 Stage 1 candidates. It uses engineered behavioral features (like budget alignment and category affinity) to re-order the candidates, outputting the final Top-12 recommendations.



#### **Key Engineering Challenges**



* **Memory Optimization:** Implemented aggressive downcasting of pandas data types (excluding IDs to preserve leading zeroes) and strategic garbage collection (gc.collect()) to process 1.3M customers within local RAM constraints.



* **Train-Serve Skew Mitigation:** Enforced strict left joins during candidate generation to ensure the LightGBM model was only trained on the exact retrieval flags it would see during production inference.



* **Negative Sampling:** Handled a highly imbalanced dataset (50:1 candidate-to-purchase ratio) by structuring the data as a LambdaRank problem, forcing the model to learn relative preferences rather than binary probabilities.



#### **Model Interpretability \& Business Insights**



A core focus of this project was validating the model's logic using SHAP and feature gain importance.



**1. Dominance of Temporal Decay:** days\_since\_purchase emerged as the primary driver of the model. This reveals that in fashion, a customer’s current "shopping window" is the strongest predictor of conversion. The model uses this signal to prioritize active users, ensuring that recommendations are fresh and timely.



**2. Price DNA \& Budget Guardrails:** The high importance of price\_z\_score and price\_ratio confirms that customers adhere to strict spending brackets. As seen in the SHAP analysis, high price outliers are heavily penalized, ensuring the model acts as a "Budget Guardrail" that keeps recommendations within a user's established spending power.



**3. Cross-Category Style Affinity:** Beyond simple item-to-item matching, the model relies on department\_name and product\_type\_name\_fav. This shows the system has successfully moved toward "Style Affinity"—identifying users who shop specifically within certain "sections" (e.g., Divided, Trend, or Premium Quality) rather than just looking at individual product IDs.



**4. Dual-Signal Model Logic (SHAP Insight):** The SHAP beeswarm confirms a sophisticated balance in the ranking logic. While behavioral features like price act as rigorous negative filters (filtering out irrelevant items), the Stage 1 heuristics (e.g., from\_seasonal, from\_assoc) provide strong positive pushes. This proves the model is acting as both an elevator of high-intent trends and a guardian of user preferences.



#### **Tech Stack**



* **Language:** Python



* **Data Processing:** Pandas, NumPy



* **Modeling:** LightGBM (LGBMRanker)



* **Interpretability \& Visualization:** SHAP, Matplotlib, Seaborn
