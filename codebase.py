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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
import seaborn as sns


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

def find_optimal_num_strata(df, use_features, max_clusters=10):
    """
    Find the optimal number of strata based on minimizing the total within-cluster variance.

    :param df: DataFrame with features and target 'y'
    :param max_clusters: Maximum number of clusters to test
    :param use_features: Flag to decide whether to use all features or mutual information
    :return: optimal number of clusters
    """
    X = df.drop(columns=['y'])  # Features
    y = df['y']                 # Target variable

    # Step 1: Optionally calculate mutual information between features and target
    mi_scores = mutual_info_classif(X, y, random_state=42)  # Uncomment this if you want to use MI
    df['mutual_info'] = np.dot(X.values, mi_scores)  # Weighted MI for each row  # Uncomment this if using MI

    best_num_clusters = 1
    min_variance = np.inf  # Initialize with a large value

    # Step 2: Try different numbers of clusters (strata) and compute the within-stratum variance
    for n_clusters in range(2, max_clusters + 1):  # Try from 2 to max_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        if use_features:
            # Stratify based on all features
            df['stratum'] = kmeans.fit_predict(X)  # Use all features for stratification
        else:
            # Stratify based on mutual information (MI) if desired
            df['stratum'] = kmeans.fit_predict(df[['mutual_info']])  # Use MI for stratification
            # pass  # MI-based stratification is commented out

        # Step 3: Calculate the within-cluster variance for the target variable 'y'
        stratum_variances = df.groupby('stratum')['y'].var()
        total_variance = stratum_variances.sum()

        print("Stratum="+str(n_clusters)+", total var:", total_variance)

        # If the total variance is lower, update the best number of clusters
        if total_variance < min_variance:
            min_variance = total_variance
            best_num_clusters = n_clusters

    return best_num_clusters

def allocation_function(df, variant):
    # Neyman Allocation
    if variant=="Neyman":
        print("Neyman Allocation")
        # Step 4: Perform undersampling with Neyman allocation
        undersampled_dfs = []
        total_minority_samples = len(df[df['y'] == 1])

        # Calculate total variance-weighted size
        stratum_stats = df.groupby('stratum').agg(
            N=('y', 'size'),
            S=('y', 'std')
        ).dropna()

        stratum_stats['weighted'] = stratum_stats['N'] * stratum_stats['S']
        total_weight = stratum_stats['weighted'].sum()

        print(total_minority_samples)
        print(stratum_stats['weighted'])
        print(total_weight)

        # Neyman allocation for each stratum
        stratum_stats['sample_size'] = (
            total_minority_samples * stratum_stats['weighted'] / total_weight
        ).astype(int)

        print(stratum_stats['sample_size'])

        # Apply undersampling per stratum
        for stratum, group in df.groupby('stratum'):
            majority = group[group['y'] == 0]
            minority = group[group['y'] == 1]

            # Neyman allocated sample size for majority class
            n_samples = stratum_stats.loc[stratum, 'sample_size']

            # print(n_samples)

            # Undersample the majority class
            undersampled_majority = resample(
                majority,
                # replace=False,
                replace=True,
                n_samples=min(n_samples, len(majority)),  # Handle cases where n_samples > available
                random_state=42
            )

            # Combine with the minority class
            undersampled_dfs.append(pd.concat([undersampled_majority, minority]))

    # Optimal Allocation
    elif variant=="Optimal":
        print("Optimal Allocation")
        # Step 4: Perform undersampling with optimal allocation
        undersampled_dfs = []
        total_minority_samples = len(df[df['y'] == 1])

        # Step 4.1: Calculate statistics and cost-adjusted weights
        stratum_stats = df.groupby('stratum').agg(
            N=('y', 'size'),
            S=('y', 'std')
        ).dropna()

        factor=5

        # Define non-uniform cost (penalize larger strata)
        stratum_stats = stratum_stats.sort_values(by='N')
        stratum_stats['C'] = factor ** (stratum_stats.index.to_series().rank() - 1)  # Costs: 1, 10, 100, ...

        stratum_stats['weighted'] = stratum_stats['N'] * stratum_stats['S'] / stratum_stats['C']
        total_weight = stratum_stats['weighted'].sum()

        # Calculate optimal allocation sample sizes
        stratum_stats['sample_size'] = (
            total_minority_samples * stratum_stats['weighted'] / total_weight
        ).astype(int)

        # Step 4.2: Apply undersampling per stratum
        for stratum, group in df.groupby('stratum'):
            majority = group[group['y'] == 0]
            minority = group[group['y'] == 1]

            # Include all minority class samples
            n_samples_minority = len(minority)

            # Check if there are no minority class samples
            if n_samples_minority == 0:
                # If no minority class, do regular resampling for majority class
                n_samples_majority = stratum_stats.loc[stratum, 'sample_size']
                
                # Regular resampling from the majority class
                undersampled_majority = resample(
                    majority,
                    # replace=False,
                    replace=True,
                    n_samples=n_samples_majority,  # Regular sample size for majority class
                    random_state=42
                )

                # Add only the resampled majority class (since there are no minority class samples)
                undersampled_dfs.append(undersampled_majority)
            else:
                # Otherwise, perform undersampling with optimal allocation
                n_samples_majority = stratum_stats.loc[stratum, 'sample_size']

                # If n_samples_majority becomes 0, we can either skip the stratum or handle it differently
                if n_samples_majority > 0:
                    # Undersample the majority class
                    undersampled_majority = resample(
                        majority,
                        # replace=False,
                        replace=True,
                        n_samples=min(n_samples_majority, len(majority)),  # Handle cases where n_samples > available
                        random_state=42
                    )

                    # Combine all minority samples with the undersampled majority class
                    undersampled_dfs.append(pd.concat([undersampled_majority, minority]))
                else:
                    # Handle case where n_samples_majority == 0 (e.g., skip or add all available samples)
                    undersampled_dfs.append(minority)  # Just append the minority class samples

    # other allocation technique: assume no allocation
    else:
        print("NO Allocation")
        # Step 4: Perform undersampling through stratified SRS
        undersampled_dfs = []
        for stratum, group in df.groupby('stratum'):
            # Separate the majority and minority class in the group
            majority = group[group['y'] == 0]
            minority = group[group['y'] == 1]
            
            # Undersample the majority class
            undersampled_majority = resample(
                majority,
                # replace=False,
                replace=True,
                n_samples=len(minority),  # Match minority class size
                random_state=42
            )
            
            # Combine with the minority class
            undersampled_dfs.append(pd.concat([undersampled_majority, minority]))
        
    return undersampled_dfs


def sampling_through_mutual_information(df, num_strata_fixed, use_features, allocation, max_clusters=10):
    """
    Perform stratified undersampling with optional dynamic determination of the number of strata.
    
    :param df: DataFrame with features and target 'y'.
    :param variant: 0 for a fixed number of strata (default 5), 1 for optimal strata determination.
    :param max_clusters: Maximum number of clusters to test when finding optimal strata.
    :return: Balanced DataFrame with undersampled majority class.
    """


    # Step 1: Optionally calculate mutual information
    X = df.drop(columns=['y'])  # Features
    y = df['y']                 # Target variable

    mi_scores = mutual_info_classif(X, y, random_state=42)
    df['mutual_info'] = np.dot(X.values, mi_scores)  # Weighted MI for each row    

    # Step 2: Determine the number of strata
    if not num_strata_fixed:
        print("NON-FIXED strata")
        if use_features==True:
            # Use find_optimal_num_strata to determine the optimal number of clusters
            print("optimal strata via all features (variant 1A)")
            optimal_num_strata = find_optimal_num_strata(df, use_features=True, max_clusters=max_clusters)
            print(f"Optimal number of strata determined: {optimal_num_strata}")
        else: 
            # Use find_optimal_num_strata to determine the optimal number of clusters
            print("optimal strata via MI (variant 1B)")
            optimal_num_strata = find_optimal_num_strata(df, use_features=False, max_clusters=max_clusters)
            print(f"Optimal number of strata determined: {optimal_num_strata}")
    else:
        # Use default fixed number of clusters
        print("FIXED strata")
        optimal_num_strata = 5
        print(f"Using fixed number of strata: {optimal_num_strata}")

    # Step 3: Group samples into stratums using KMeans clustering
    kmeans = KMeans(n_clusters=optimal_num_strata, random_state=42)
    df['stratum'] = kmeans.fit_predict(df[['mutual_info']])

    # Step 4: Perform undersampling through stratified SRS (within an allocation function)
    undersampled_dfs=allocation_function(df, variant=allocation) 

    # Step 5: Combine sampled stratums
    final_df = pd.concat(undersampled_dfs)

    # Shuffle the resulting dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop unneeded columns
    final_df.drop(columns=['mutual_info', 'stratum'], inplace=True)

    print(final_df['y'].value_counts())

    return final_df


# NOTE: Only put one sampling function at a time
def sampling(df):
    """
    Undersample through maximizing mutual information: (3 parameters 12 VARIANTS)

    parameter 1: num_strata_fixed; is the number of strata fixed or not (2 values: True/ False)
    parameter 2: use_features; either use the data's features or mutual information to determine the number of strata IF num_strata_fixed is true (2 values: True/False)
    parameter 3: allocation; the allocation type use to sample data from each stratum (3 values: 'Neyman', 'Optimal', None)
    """

    # Alex 
    output=sampling_through_mutual_information(df, num_strata_fixed=False, use_features=False, allocation=" ")

    # ADD YOUR WAY TO RESAMPLE THE DATA...

    # # Shubham
    # output=sampling_through_support_point()

    # # Yolanda 
    # output=sampling_through_principal_point()

    return output

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
    
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
def baseline_testing(df):

    # first do classiciation in baseline (data without resample)
    classification_and_evaluation(df)

    # baseline testing on 3 undersampling techniques (random undersampling, NearMiss, TomekLinks)
    classification_and_evaluation(df, baseline="R")
    classification_and_evaluation(df, baseline="N")
    classification_and_evaluation(df, baseline="T")


def classification_and_evaluation(df, baseline=None):
    X = df.drop(columns=['y'])  # Features
    y = df['y']   

    # set random seed for code reproducability

    # 1. data splitting into train-test-validation split (80-10-10)
    # first split train as 80%, rest 20%
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    print(X_train.shape)

    # # split the remaining 20% by half (10% test, 10% validation)
    # X_val, X_test, y_val, y_test=train_test_split(X_test,y_test,test_size=0.5,random_state=7)

# specifically for baseline
    if baseline:
        print("BASELINE!")

        # For Random undersampling
        if baseline[0]=='R':
            # Initialize RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            # Apply random undersampling
            X_train, y_train = rus.fit_resample(X_train, y_train)
            print(X_train.shape)

        # For NearMiss
        elif baseline[0]=='N':
            # Initialize NearMiss
            nm = NearMiss(version=1)  # You can experiment with version=2 or 3 for different strategies

            # Apply NearMiss undersampling
            X_train, y_train = nm.fit_resample(X_train, y_train)
            print(X_train.shape)
        
        # For Tomek Links
        elif baseline[0]=='T':
            # Initialize TomekLinks
            tl = TomekLinks()

            # Apply Tomek Links undersampling
            X_train, y_train = tl.fit_resample(X_train, y_train)
            print(X_train.shape)


    # 2. train and fit a decision tree model using the training data
    RF = RandomForestClassifier(random_state=42)
    RF.fit(X_train, y_train)

    # 2.1 make predictions on the testing data based on the pre-tuned model fitted with training data
    RF_y_train_pred=RF.predict(X_test)
    # Accuracy Score, they are around 85-90%
    print("Training (Pre-tuned) Accuracy on Decision Tree: {}".format(accuracy_score(y_test, RF_y_train_pred)))

    # print evaluation
    evaluate_model(y_test, RF_y_train_pred)

from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
def preprocess_breast_cancer_data():
    # fetch dataset 
    breast_cancer = fetch_ucirepo(id=14) 
    
    # data (as pandas dataframes) 
    X = breast_cancer.data.features 
    y = breast_cancer.data.targets 

    df = pd.concat([X, y], axis=1)

    # rename values for inv-nodes and tumor-size
    df['inv-nodes']=df['inv-nodes'].replace({'5-Mar':'3-5', '8-Jun':'6-8', '11-Sep':'9-11', '14-Dec':'12-14'})
    df['tumor-size']=df['tumor-size'].replace({'14-Oct':'10-14', '9-May':'5-9'})

    # handle null data
    df = df.fillna(df.mode().iloc[0])

    # drop duplciates
    df = df.drop_duplicates() # dropped 14 (286 to 272)


    # Data transfromation
    label_encoder=LabelEncoder()

    df_copy=df.copy()

    alist=['irradiat','node-caps', 'Class']
    for i in alist:
        df_copy[i]=label_encoder.fit_transform(df_copy[i])

    # ordinal for menopause, can try the range ones
    df_copy['menopause']=df_copy['menopause'].map({"lt40": 0, "ge40": 1, "premeno": 2})
    df_copy['tumor-size']=df_copy['tumor-size'].map({"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10, "55-59": 11})
    df_copy['inv-nodes']=df_copy['inv-nodes'].map({"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8, "27-29": 9, "30-32": 10, "33-35": 11, "36-39": 12})
    df_copy['age']=df_copy['age'].map({"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4, "60-69": 5, "70-79": 6, "80-89": 7, "90-99": 8})

    df=pd.get_dummies(df_copy, columns=['breast', 'breast-quad'])

    df.rename(columns={'Class':'y'}, inplace=True)
    # print(df.columns)

    return df


# MAIN FUNCTION
def main():

    # BREAST CANCER DATA PART STARTS------------------------
    breast_cancer_data_path="https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/processed_breast_cancer_data.csv"

    # option 1: start from the beginning
    # breast_cancer_data=preprocess_breast_cancer_data()
    # option 2: load the processed data file from github repo (USE THIS OPTION)
    # breast_cancer_data=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/processed_breast_cancer_data.csv")
    # ONCE LOADED, JUST APPLY data same as below

    # BREAST CANCER DATA PART ENDS---------------------------


    baseline_data_path="https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/Processed_Input.csv"

    # baseline testing
    baseline_testing(pd.read_csv(baseline_data_path))

    # EXPERIMENT STARTS
    # Step 1 of the project: data collection and preprocessing (CHOOSE ONE OF THE 2)

    # option 1: start from the beginning
    # data=data_collection_and_preprocessing()
    # option 2: load the processed data file from github repo (USE THIS OPTION)
    data=pd.read_csv(baseline_data_path)

    print(data['y'].value_counts())

    # step 2: sampling (ONLY DIFFERENCE)
    resampled_data=sampling(data)
    
    # # step 3: do classification (and prediction)

    # then do classification with the MI approach
    classification_and_evaluation(resampled_data)

main()
