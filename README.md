# CIC-IDS 2017 Network Intrusion Detection Dataset Analysis

A comprehensive machine learning project for analyzing and detecting network intrusions using the CIC-IDS 2017 dataset. This project implements 12 state-of-the-art algorithms for cybersecurity threat detection and includes forensic investigation capabilities.

## 📊 Project Overview

The CIC-IDS 2017 dataset contains network traffic data captured over 5 days, including both benign and malicious activities. This project provides:

- **Advanced Data Analysis**: Comprehensive exploration of network flow features
- **Machine Learning Models**: Implementation of 12 different algorithms for intrusion detection
- **Forensic Investigation**: Professional cybersecurity analysis and reporting
- **Model Interpretability**: SHAP and LIME analysis for explainable AI

## 🎯 Dataset Description

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

## 🚀 Features

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis
- **Feature Engineering**: Advanced feature selection and dimensionality reduction
- **Data Visualization**: Interactive plots using Matplotlib, Seaborn, and Plotly
- **Data Quality Assessment**: Missing value analysis and data cleaning

### Machine Learning Algorithms
1. **Random Forest Classifier**
2. **XGBoost**
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

### Advanced Analytics
- **Model Interpretability**: SHAP values and LIME explanations
- **Ensemble Methods**: Voting and Stacking classifiers
- **Hyperparameter Tuning**: GridSearch and RandomizedSearch
- **Cross-Validation**: Robust model evaluation
- **ROC Analysis**: Performance metrics and visualization

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

## 🛠️ Installation

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
   - Visit the [official CIC-IDS 2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html)
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
2. Add the CIC-IDS 2017 dataset to your Kaggle notebook
3. Run the notebooks with automatic environment detection

## 📊 Key Results

- **Detection Accuracy**: Achieved >99% accuracy with ensemble methods
- **Feature Importance**: Identified critical network flow characteristics
- **Attack Classification**: Successfully classified 15 different attack types
- **Real-time Capability**: Optimized models for production deployment

## 🔍 Forensic Investigation

The project includes comprehensive forensic analysis:

- **Attack Pattern Analysis**: Detailed examination of malicious traffic
- **Timeline Reconstruction**: Sequential analysis of attack events
- **Professional Reporting**: Industry-standard cybersecurity documentation
- **Evidence Preservation**: Proper handling of digital evidence

## 📈 Performance Metrics

- **Precision**: Attack detection accuracy
- **Recall**: Coverage of malicious activities
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Model discrimination capability
- **Confusion Matrix**: Detailed classification results

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