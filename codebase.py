# Load the dataset as csv file as dataframe (df)

# For part 1 (data collection and preprocessing)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For part 2 (undersampling via stratified SRS)
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.utils import resample

# For part 3 (classificaiton and prediction)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state


def data_transformation(data):
    # Data Transformation

    # 1. Encode binary variables
    # Initialize LabelEncoder for binary encoding
    label_encoder = LabelEncoder()

    # 1.1 encode binary predictors
    data['default']=label_encoder.fit_transform(data['default'])
    data['housing']=label_encoder.fit_transform(data['housing'])
    data['loan']=label_encoder.fit_transform(data['loan'])

    # 1.2 encode binary target variable (y)
    data['y']=label_encoder.fit_transform(data['y'])

    # 2. encoding categorical data with multiple class (job, marital, education, contact, month, poutcome)

    # 2.1 month: since months are just the abbreviations (short forms of the month) --> encode strings to months numerically
    data['month']=label_encoder.fit_transform(data['month'])+1

    # 2.2 education (unknown exists): map values with setted values where higher educations are mapped with higher values
    data['education']=data['education'].map({"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3})

    # 2.3 poutcome (unknown exists): map values with setted values where failure is 0, success is 1, unknown and otheras 0.5
    data['poutcome']=data['poutcome'].map({"failure": 0, "success": 1, "unknown": 0.5, "other": 0.5})

    # I decided to leave the rest untouched because tree-based methods can handle categorical data pretty well already.
    # Hence I did not handle: contact, marital, and job (as of now)

    # 3. replace unknown with meaningful value (for contact, education, job)

    # 3.1 job, try to categorize retired and student out first based on age, then see what to do
    # everyone under 22 are already students, so transform elders
    data.loc[data['age'] >66, 'job'] = data.loc[data['age'] >66, 'job'].replace('unknown', 'retired')
    # print(data['job'].value_counts()) # 278 unknown values now, 10 unknown ones are removed

    # 3.2: contact, replace with mode--> cellular
    # cellular returns 65%, unknown returns 29%, telephone returns 6%
    data['contact']=data['contact'].replace("unknown", "cellular")

    # 3.3 education, replace with mode--> 2
    # 0: 4%, 1: 15%, 2: 51%, 3: 29%
    data['education']=data['education'].replace(0, 2)

    # One-hot encode 'contact', 'marital', 'job'
    data=pd.get_dummies(data, columns=['contact', 'marital', 'job'])

    return data

def data_collection_and_preprocessing():
    # # Read travel data
    # df=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/travel.csv")

    # Read bank telemarketing dataset
    df=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/input.csv", sep=";")

    # print the proportion of both classes
    print("\nPercentage of total samples in NO:",df['y'].value_counts()[0]/df.shape[0]*100)
    print("Percentage of total samples in YES:",df['y'].value_counts()[1]/df.shape[0]*100)

    # do data transformation 
    processed_df=data_transformation(df)
    return processed_df

def sampling_through_mutual_informatinon(df):
    X = df.drop(columns=['y'])  # Features
    y = df['y']                 # Target variable

    # Step 1: Calculate mutual information
    mi_scores = mutual_info_classif(X, y)
    df['mutual_info'] = np.dot(X.values, mi_scores)  # Weighted MI for each row    

    # Step 2: Group samples into stratums (e.g., using k-means clustering)
    kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust number of stratums
    df['stratum'] = kmeans.fit_predict(df[['mutual_info']])

    # Step 3: Perform undersampling through stratified SRS
    undersampled_dfs = []
    for stratum, group in df.groupby('stratum'):
        # Separate the majority and minority class in the group
        majority = group[group['y'] == 0]
        minority = group[group['y'] == 1]
        
        # Undersample the majority class
        undersampled_majority = resample(
            majority,
            replace=False,
            n_samples=len(minority),  # Match minority class size
            random_state=42
        )
        
        # Combine with the minority class
        undersampled_dfs.append(pd.concat([undersampled_majority, minority]))

    # Step 4: Combine sampled stratums
    final_df = pd.concat(undersampled_dfs)

    # Shuffle the resulting dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # drop unneeded column
    final_df.drop(columns=['mutual_info', 'stratum'], inplace=True)

    return final_df

# NOTE: Only put one sampling function at a time
def sampling(input):
    # undersample through maximizing MI 
    output=sampling_through_mutual_informatinon(input)

    # ADD YOUR WAY TO RESAMPLE THE DATA...

    return output

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
import seaborn as sns

def evaluate_model(y_true, y_pred):
    """
    Evaluates a classification model by printing the confusion matrix, 
    classification report, class-wise metrics, and plotting the AUC-ROC 
    and precision-recall curves.
    
    Parameters:
        y_true (array-like): Actual labels
        y_pred (array-like): Predicted probabilities or labels
    """
    # Handle binary or probabilistic predictions
    y_pred_labels = np.where(y_pred >= 0.5, 1, 0) if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_labels)
    print(cm)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # plt.savefig("RF_confusion_matrix.png")


    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels))

    # # Class-wise Performance Metrics
    # print("\nClass-wise Performance Metrics:")
    # for class_label in range(cm.shape[0]):
    #     tp = cm[class_label, class_label]
    #     fn = cm[class_label].sum() - tp
    #     fp = cm[:, class_label].sum() - tp
    #     tn = cm.sum() - (tp + fn + fp)
    #     print(f"Class {class_label} -> TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

    # Class-wise Performance Metrics
    classification_report_2 = classification_report(y_true, y_pred_labels, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {metric: [classification_report_2[class_name][metric] for class_name in ['0','1']] for metric in metrics}
    plt.figure(figsize=(10, 6))
    sns.heatmap(data=pd.DataFrame(class_metrics, index=['No','Yes']), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Class-wise Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    # plt.savefig("RF_performance_metrics.png")
    plt.show()

    
    # AUC-ROC Curve
    if len(np.unique(y_true)) == 2:  # Binary classification
        roc_auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUC-ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_pred)
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()





def classification_and_evaluation(df):
    X = df.drop(columns=['y'])  # Features
    y = df['y']   

    # set random seed for code reproducability
    random_state=check_random_state(42)

    # 1. data splitting into train-test-validation split (80-10-10)
    # first split train as 80%, rest 20%
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=random_state)

    # # split the remaining 20% by half (10% test, 10% validation)
    # X_val, X_test, y_val, y_test=train_test_split(X_test,y_test,test_size=0.5,random_state=7)

    # 2. train and fit a decision tree model using the training data
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)

    # 2.1 make predictions on the testing data based on the pre-tuned model fitted with training data
    RF_y_train_pred=RF.predict(X_test)
    # Accuracy Score, they are around 85-90%
    print("Training (Pre-tuned) Accuracy on Decision Tree: {}".format(accuracy_score(y_test, RF_y_train_pred)))

    # print evaluation
    evaluate_model(y_test, RF_y_train_pred)


def main():
    # Step 1 of the project: data collection and preprocessing (CHOOSE ONE OF THE 2)

    # option 1: start from the beginning
    # data=data_collection_and_preprocessing()
    # option 2: load the processed data file from github repo
    data=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/Processed_Input.csv")

    # step 2: sampling 
    resampled_data=sampling(data)
    
    # step 3: do classification (and prediction)

    # first do classiciation in baseline 
    classification_and_evaluation(pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/Processed_Input.csv"))
    # then do classification with the MI approach
    classification_and_evaluation(resampled_data)
    
main()


