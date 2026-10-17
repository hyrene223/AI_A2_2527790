# Responsible AI and Ethical Use

## 1. Intended Use
Musanze Harvest Intelligence Service is an educational machine-learning project that demonstrates crop-yield prediction, high-risk classification, and clustering. It is designed for learning, experimentation, and demonstration using synthetic data.

## 2. Data Source
The dataset is synthetically generated for this assignment. It does not represent verified measurements from actual farms or farmers. Model results therefore cannot establish real-world farming performance.

## 3. Limitations
- Synthetic data may not reflect real weather, soil, crop varieties, pests, farming practices, or local conditions.
- Model performance depends on the assumptions used to generate the data.
- The dataset is relatively small and cannot represent all farming environments.
- A high test score does not guarantee reliable predictions on real farms.
- Clustering describes patterns in the generated data, not verified categories of farmers or farms.

## 4. Data Leakage Prevention
The regression target, yield_tons, is not used as an input feature. The high_risk target is excluded from classification inputs. Preprocessing is fitted on training data only, and evaluation uses a separate test set.

## 5. Fairness and Potential Harm
Predictions may be inaccurate for conditions that are not represented in the generated data. Using them without validation could lead to poor farming, financial, or resource-allocation decisions.

## 6. Privacy and Security
The project uses synthetic records and does not require personal farmer information. If adapted to real data, appropriate consent, access controls, data minimization, and secure storage should be implemented.

## 7. Human Oversight and Prohibited Uses
Do not use this prototype as the sole basis for agricultural loans, insurance decisions, government assistance, land allocation, or other high-impact decisions. Do not treat its risk predictions as confirmed diagnoses of farm conditions. Real-world use requires suitable field data, independent validation, expert review, and ongoing monitoring.

## 8. Responsible Conclusion
This project demonstrates machine-learning workflows rather than a production-ready agricultural decision system. Users should interpret outputs cautiously and seek advice from qualified agricultural experts before making real farming decisions.
