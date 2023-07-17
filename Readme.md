# Breast Cancer Diagnosis Prediction
This project is about unlocking the power of machine learning to aid in the early detection of breast cancer. It revolves around the development of a predictive model that can accurately determine whether a breast tumor is malignant, which means it's cancerous, or benign, signifying it's not cancerous.

Leveraging the [Breast Cancer Wisconsin (Diagnostic) Data Set- Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) encompassing various characteristics of breast tumors, the aim here is not only to potentially enhance healthcare outcomes for patients but also to illustrate the tangible benefits machine learning can bring to the medical field.

Through the Python-based script in this project, we take a deep dive into the dataset, analyze and visualize the data, and then step-by-step, build a Logistic Regression model - a popular algorithm in the realm of machine learning. 
By the end of this project, we are able to make predictions about tumor malignancy with impressive accuracy, demonstrating the real-world application and impact of machine learning in healthcare.

## Dataset
The dataset used for this project is breast_cancer_dataset.csv, which contains information about breast tumors. Each row in the dataset corresponds to a different tumor, and each column represents a different feature of the tumor, like the size, shape, and so on. The 'diagnosis' column indicates whether the tumor was malignant (M) or benign (B).

## The Code
The script breast_cancer_prediction.py is a step-by-step guide for training and evaluating a Logistic Regression model to predict the diagnosis of breast tumors.

**Here's a summary of what the script does:**

1. Import Libraries: The script starts by importing necessary Python libraries for data manipulation, visualization, and machine learning.
2. Load & Explore the Dataset: It then loads the dataset using pandas, and explores the structure and features of the dataset.
3. Data Preprocessing: The script performs some basic data cleaning, such as removing unnecessary columns and converting categorical data into numerical form.
4. Data Visualization: The script creates box plots of selected features against the diagnosis to see how these features differ between malignant and benign tumors.
5. Split the Data: The data is split into a training set and a testing set. The training set is used to train the model, and the testing set is used to evaluate the model's performance.
6. Feature Scaling: The features are scaled so that they all have a similar range. This is important for many machine learning models.
7. Build Logistic Regression Classifier: The script creates a Logistic Regression model, which is a popular machine learning model for binary classification tasks.
8. Evaluate Model Performance: Finally, the script evaluates the performance of the model on the testing data, using metrics like accuracy, precision, recall, and F1 score. It also outputs a confusion matrix, which shows the number of true positives, true negatives, false positives, and false negatives.

## How to Run the Code
1. Ensure you have Python installed on your computer. You will also need the following Python libraries: pandas, numpy, sklearn, matplotlib, seaborn. You can install these using pip: `pip install pandas numpy sklearn matplotlib seaborn`.
2. Download the breast_cancer_prediction.py script and the breast_cancer_dataset.csv dataset and put them in the same folder.
3. Run the script using a Python interpreter. If you're using a terminal or command line, navigate to the folder containing the script and dataset, and run `python breast_cancer_prediction.py`.
4. This project demonstrates a basic workflow for training and evaluating a machine learning model using Python and sci-kit-learn. **The resulting model can predict whether a breast tumor is malignant or benign with reasonable accuracy.** However, it's important to remember that real-world applications of machine learning in healthcare would require much more rigorous validation and testing before use.

## Related Links
* [Breast Cancer Wisconsin (Diagnostic) Data Set- Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* [Related Article by the National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8993572/)
* [Simple AI - Newsletter](https://www.linkedin.com/build-relation/newsletter-follow?entityUrn=7041344488937541632)
