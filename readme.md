# **Loan Approval Prediction using Machine Learning**

This project is a machine learning model designed to automate the loan approval process. By analyzing applicant data, it predicts whether a loan application should be approved (Y) or rejected (N). The goal is to create an intelligent system that increases efficiency, accuracy, and fairness, reducing the time and potential human bias involved in manual loan assessments.

This project was developed by **Gourav Acharjee** (Reg. No: 2212701110123, Roll No: 430122010197\) as part of the B.Tech in Computer Science & Engineering curriculum at Narula Institute of Technology.

## **📋 Table of Contents**

* [Project Aim](#bookmark=id.nws8f65x1ruj)  
* [Workflow](#bookmark=id.c36vnmu0zkih)  
* [Dataset](#bookmark=id.74ozuilc809y)  
* [Models Implemented](#bookmark=id.bjxunocljtyn)  
* [Results](#bookmark=id.tb5e421z8syp)  
* [Technologies Used](#bookmark=id.kiht20ih2mlc)  
* [How to Run](#bookmark=id.6votpcom4hhl)  
* [Limitations](#bookmark=id.vcmv221gd9cg)  
* [Future Scope](#bookmark=id.v3ppy3g75gmb)  
* [Acknowledgments](#bookmark=id.ko6etxx6ljr2)

## **🎯 Project Aim**

The primary objectives of this project are:

* To study and analyze the different parameters that influence loan approval (e.g., credit history, income, education).  
* To clean, preprocess, and analyze the dataset to extract meaningful insights.  
* To apply, train, and compare the performance of various machine learning algorithms for this classification task.  
* To develop a reliable system that can accurately classify loan applications, helping financial institutions minimize loan default risks.  
* To automate the assessment procedure to reduce human effort and processing time.

## **⚙️ Workflow**

The project follows a standard machine learning pipeline:

1. **Data Collection:** The dataset (loan.csv) is loaded, containing applicant financial and demographic attributes.  
2. **Data Preprocessing:**  
   * Missing values are handled (e.g., dropna()).  
   * Categorical features (like Gender, Married, Property\_Area, etc.) are converted into numerical values using LabelEncoder.  
   * Irrelevant features (like Loan\_ID) are dropped.  
3. **Exploratory Data Analysis (EDA):**  
   * Visualizations are created using matplotlib and seaborn (e.g., countplot, pairplot).  
   * This step helps to understand the distribution of data and the relationship between different features and the Loan\_Status target variable.  
4. **Data Splitting:** The preprocessed dataset is split into training and testing sets (80% train, 20% test).  
5. **Model Training:** Multiple classification algorithms are trained on the training data.  
6. **Model Evaluation:** The trained models are evaluated on the test set using standard metrics:  
   * Accuracy Score  
   * Confusion Matrix  
   * Classification Report (Precision, Recall, F1-Score)  
7. **Model Selection:** The best-performing model is identified for potential deployment.

## **🗂️ Dataset**

The dataset contains the following features used for prediction:

* Gender  
* Married  
* Dependents  
* Education  
* Self\_Employed  
* ApplicantIncome  
* CoapplicantIncome  
* LoanAmount  
* Loan\_Amount\_Term  
* Credit\_History  
* Property\_Area  
* Loan\_Status (Target Variable: Y/N)

## **🤖 Models Implemented**

Several machine learning models were trained and compared:

* K-Nearest Neighbors (KNN)  
* Logistic Regression  
* Naive Bayes (GaussianNB)  
* Support Vector Machine (SVM \- Linear Kernel)  
* Decision Tree Classifier  
* Random Forest Classifier

## **📊 Results**

The performance of the models on the test set was as follows:

| Model | Accuracy Score |
| :---- | :---- |
| **Support Vector Machine (Linear)** | **84.4%** |
| Logistic Regression | 82.3% |
| Random Forest Classifier | 80.2% |
| K-Nearest Neighbors (KNN) | 78.1% |
| Naive Bayes (GaussianNB) | 78.1% |
| Decision Tree Classifier | 70.8% |

Based on the evaluation, the **Support Vector Machine (SVM) with a linear kernel** provided the highest accuracy for this specific dataset and preprocessing workflow.

## **🛠️ Technologies Used**

* **Python 3.7+**  
* **Libraries:**  
  * Pandas (for data manipulation and analysis)  
  * NumPy (for numerical operations)  
* **Machine Learning:**  
  * Scikit-learn (for model implementation, splitting, and evaluation)  
* **Data Visualization:**  
  * Matplotlib  
  * Seaborn  
* **Environment:**  
  * Jupyter Notebook / Google Colab

## **🚀 How to Run**

1. **Clone the repository:**  
   git clone \[https://github.com/your-username/loan-approval-prediction.git\](https://github.com/your-username/loan-approval-prediction.git)  
   cd loan-approval-prediction

2. **Install the required libraries:**  
   pip install pandas numpy matplotlib seaborn scikit-learn

3. **Run the Jupyter Notebook:**  
   * Open the .ipynb file (e.g., loan\_prediction.ipynb) in Jupyter Notebook, Jupyter Lab, or Google Colab.  
   * Ensure the loan.csv dataset file is in the same directory.  
   * Run the cells sequentially to see the preprocessing, analysis, training, and evaluation steps.

## **⚠️ Limitations**

As noted in the project report, the current system has some limitations:

* **Dataset Size:** The dataset is relatively small and may not capture the full diversity of all loan applicants.  
* **Static Data:** The model is trained on a static dataset and does not update with new or real-time data.  
* **Basic Models:** The project focuses on traditional ML models; more advanced techniques like gradient boosting (XGBoost) or deep learning were not explored.  
* **Binary Classification:** The model only predicts "Approved" or "Rejected" and does not suggest eligible loan amounts or interest rates.

## **💡 Future Scope**

Future work could enhance this project significantly:

* **Expand Dataset:** Incorporate larger and more diverse datasets from various financial institutions.  
* **Real-Time Integration:** Connect the system to live financial databases or credit bureau APIs.  
* **Advanced Models:** Implement and test gradient boosting methods (XGBoost, LightGBM) or deep learning (Neural Networks) to potentially improve accuracy.  
* **Web Deployment:** Create a web application (using Flask or Django) to provide a user-friendly interface for applicants or bank employees to use the model.  
* **Predict Loan Default:** Extend the model to not only predict approval but also the probability of a loan default.

## **🙏 Acknowledgments**

* This project was guided by **Mr. Partha Koley**, whose knowledge and constructive feedback were instrumental in its successful completion.  
* Sincere thanks to all the teachers at **Narula Institute of Technology** for their guidance and support.