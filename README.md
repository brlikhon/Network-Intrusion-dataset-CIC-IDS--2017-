# Network Intrusion Detection with CIC-IDS-2017

This project provides a comprehensive analysis of the CIC-IDS-2017 dataset to build and evaluate machine learning models for network intrusion detection. The entire pipeline, from data cleaning and exploratory analysis to model training, interpretation, and forensic investigation, is documented.

## 📊 Project Overview

The primary goals of this analysis are to:
* Analyze the CIC-IDS-2017 intrusion detection dataset
* Build and evaluate high-performance machine learning models for attack detection
* Conduct a detailed forensic investigation into identified attacks
* Develop a reproducible pipeline for network traffic analysis

This project provides:
- **Advanced Data Analysis**: Comprehensive exploration of network flow features
- **Machine Learning Models**: Implementation of 12 different algorithms for intrusion detection
- **Forensic Investigation**: Professional cybersecurity analysis and reporting
- **Model Interpretability**: SHAP and LIME analysis for explainable AI

## 🎯 Dataset Description

* **Dataset:** Canadian Institute for Cybersecurity - Intrusion Detection System 2017 (CIC-IDS-2017)
* **Description:** The dataset contains network traffic captured over 5 days, including a wide variety of benign and malicious activities
* **Size:** The data is distributed across 8 CSV files, totaling 843.7 MB
* **Content:**
    * Initial Records: 2,830,743
    * Cleaned Records: 2,574,264 (90.9% data retention)
    * Attack Types: 14 unique attack categories were analyzed

The dataset includes network traffic captured from Monday to Friday, containing:

- **Total Records**: 2,830,743 network flows
- **Features**: 79 network flow characteristics
- **Attack Types**: 15 different categories including:
  - BENIGN (Normal traffic)
  - DDoS attacks
  - Port Scan
  - Web Attacks (Brute Force, XSS, SQL Injection)
  - DoS variants (Hulk, GoldenEye, slowloris, Slowhttptest)
  - FTP-Patator & SSH-Patator
  - Bot attacks
  - Infiltration
  - Heartbleed

## 🔬 Methodology

The project follows a structured data science workflow:

1. **Data Cleaning and Preprocessing:** Handled missing values, infinite values, and removed over 256,000 duplicate records
2. **Feature Engineering:** Created new insightful features such as `Flow_Bytes_Per_Packet` and `Packet_Size_Ratio` to improve model performance
3. **Exploratory Data Analysis (EDA):** Visualized the distribution of attack types, analyzed traffic patterns by day, and examined feature correlations
4. **Optimized Feature Selection:** Used a consensus-based approach combining F-Score and Random Forest Importance to identify the most predictive features
5. **Model Training and Evaluation:** Trained and evaluated 5 different machine learning algorithms, including Random Forest, XGBoost, and LightGBM, on a sampled dataset for efficiency
6. **Model Interpretability (XAI):** Employed SHAP and Permutation Importance to understand the decisions of the best-performing model
7. **Forensic Investigation:** Conducted a deep-dive analysis into the most prevalent attack, **DoS Hulk**, to identify its unique network signature

## 🏆 Key Results

The analysis yielded high-performance models and actionable security insights.

### 📈 Model Performance

The **XGBoost** classifier was the top-performing algorithm with the following metrics on the test set:

* **Accuracy:** 99.87%
* **F1-Score:** 99.61%
* **Precision:** 99.66%
* **Recall:** 99.55%
* **ROC-AUC:** 99.99%

### 🔍 Key Findings

* **Feature Importance:** The model interpretability analysis identified `Destination Port`, `Init_Win_bytes_forward`, `Total Length of Fwd Packets`, and `Fwd Packet Length Max` as some of the most critical features for distinguishing attacks from benign traffic
* **Forensic Signature:** The "DoS Hulk" attack was characterized by an abnormally high `Max Packet Length` (10.53x higher than benign) and `Flow Duration` (5.34x higher), providing a clear signature for detection
* **Temporal Patterns:** Attack types were concentrated on specific days. For instance, the DoS Hulk attack occurred exclusively on Wednesday, while DDoS and PortScan attacks were captured on Friday

## 🛠️ Setup and Usage

1. **Environment:** The analysis was designed to be run in a cloud environment like Google Colab or Kaggle Notebooks
2. **Installation:** Install the required libraries using the provided command:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn ipykernel plotly scipy xgboost lightgbm catboost shap lime ipython
   ```
3. **Data:** Download the CIC-IDS-2017 dataset from available sources:
   - **Primary Dataset**: [Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
   - Place the CSV files in the appropriate directory as referenced in the notebook
4. **Execution:** Run the notebook cells sequentially to reproduce the entire analysis pipeline

## 🚀 Features

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis
- **Feature Engineering**: Advanced feature selection and dimensionality reduction
- **Data Visualization**: Interactive plots using Matplotlib, Seaborn, and Plotly
- **Data Quality Assessment**: Missing value analysis and data cleaning

### Machine Learning Algorithms
1. **Random Forest Classifier**
2. **XGBoost** (Best performing)
3. **LightGBM**
4. **CatBoost**
5. **Support Vector Machine**
6. **Logistic Regression**
7. **Naive Bayes**
8. **K-Nearest Neighbors**
9. **Decision Tree**
10. **AdaBoost**
11. **Gradient Boosting**
12. **Neural Networks**

## 🔧 Advanced Analytics Features

- **Model Interpretability**: SHAP values and LIME explanations
- **Ensemble Methods**: Voting and Stacking classifiers
- **Hyperparameter Tuning**: GridSearch and RandomizedSearch
- **Cross-Validation**: Robust model evaluation
- **ROC Analysis**: Performance metrics and visualization
- **Feature Engineering**: Custom feature creation for improved performance
- **Data Quality Assessment**: Comprehensive missing value and duplicate analysis

## 🎯 Practical Applications

The findings and models from this project can be used to:

* Deploy a real-time intrusion detection system (IDS)
* Prioritize network monitoring efforts based on identified key features
* Inform incident response and forensic analysis protocols

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
ipykernel>=6.0.0
plotly>=5.0.0
scipy>=1.7.0
xgboost>=1.7.0
lightgbm>=4.0.0
catboost>=1.2.0
shap>=0.41.0
lime>=0.2.0
ipython>=8.0.0
```

## 📁 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/brlikhon/Network-Intrusion-dataset-CIC-IDS--2017-.git
cd Network-Intrusion-dataset-CIC-IDS--2017-
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the CIC-IDS 2017 dataset**:
   - **Kaggle**: [Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
   - Extract CSV files to the project directory

## 📁 Project Structure

```
CIC-IDS-2017/
├── README.md                                          # Project documentation
├── requirements.txt                                   # Python dependencies
├── Network Intrusion dataset(CIC-IDS- 2017).ipynb   # Main analysis notebook
├── Twelve Algorithms Notebook.ipynb                  # 12 ML algorithms implementation
├── Forensic Investigation Report.pdf                 # Professional cybersecurity report
└── *.csv                                            # Dataset files (to be downloaded)
```

## 🚀 Usage

### Jupyter Notebook Environment

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Open the main notebook**:
   - `Network Intrusion dataset(CIC-IDS- 2017).ipynb` - Main analysis
   - `Twelve Algorithms Notebook.ipynb` - Algorithm implementations

### Kaggle Environment

The notebooks are optimized for both local and Kaggle environments:

1. Upload the project to Kaggle
2. Add the CIC-IDS 2017 dataset to your Kaggle notebook from available sources:
   - [Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
3. Run the notebooks with automatic environment detection

### 📊 Key Results

- **Detection Accuracy**: Achieved >99% accuracy with ensemble methods
- **Feature Importance**: Identified critical network flow characteristics
- **Attack Classification**: Successfully classified 15 different attack types
- **Real-time Capability**: Optimized models for production deployment

## 🔍 Forensic Investigation

The project includes comprehensive forensic analysis:

- **Attack Pattern Analysis**: Detailed examination of malicious traffic signatures
- **Timeline Reconstruction**: Sequential analysis of attack events across the 5-day period
- **DoS Hulk Deep Dive**: Specialized analysis revealing 10.53x higher packet lengths
- **Professional Reporting**: Industry-standard cybersecurity documentation
- **Evidence Preservation**: Proper handling of digital evidence for investigative purposes

## 📈 Performance Metrics

### XGBoost Model Results:
- **Accuracy**: 99.87% - Extremely high overall correctness
- **Precision**: 99.66% - Minimal false positive rate
- **Recall**: 99.55% - Excellent attack detection coverage  
- **F1-Score**: 99.61% - Balanced precision-recall performance
- **ROC-AUC**: 99.99% - Near-perfect discrimination capability

### Feature Engineering Impact:
- **Data Retention**: 90.9% after cleaning (2,574,264 records)
- **Duplicate Removal**: 256,000+ duplicate records eliminated
- **Custom Features**: Created `Flow_Bytes_Per_Packet` and `Packet_Size_Ratio`
- **Feature Selection**: Consensus-based approach using F-Score and Random Forest Importance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Canadian Institute for Cybersecurity (CIC)** for providing the dataset
- **University of New Brunswick** for cybersecurity research
- **Open Source Community** for the machine learning libraries

## 📞 Contact

- **GitHub**: [@brlikhon](https://github.com/brlikhon)
- **Project Link**: [https://github.com/brlikhon/Network-Intrusion-dataset-CIC-IDS--2017-](https://github.com/brlikhon/Network-Intrusion-dataset-CIC-IDS--2017-)

## 🔗 References

1. Sharafaldin, I., Lashkari, A.H., and Ghorbani, A.A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. ICISSP.
2. [CIC-IDS2017 Official Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

---

**⭐ Star this repository if you find it helpful!** 