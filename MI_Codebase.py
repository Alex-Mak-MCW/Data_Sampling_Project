#----------------------------------------------------------------------------------------------------------------------- 
#-----------------------------------------------------------------------------------------------------------------------
# NOTE: NO NEED TO WORRY PART STARTS

# For data collection and preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For undersampling via stratified SRS
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.utils import resample

# For classificaiton and prediction
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

# For creating a pairwise MI matrix
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Other Functions (Not impacted by experiment)
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

# NOTE: NO NEED TO WORRY PART ENDS
#----------------------------------------------------------------------------------------------------------------------- 
#-----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------- 
#-----------------------------------------------------------------------------------------------------------------------
# NOTE: EXPERIMENT PART STARTS

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
    :param num_strata_fixed: Boolean indicating whether the number of strata is fixed.
    :param use_features: Flag to decide whether to use all features or mutual information for clustering.
    :param allocation: Allocation type for undersampling (e.g., 'Neyman', 'Optimal', None).
    :param max_clusters: Maximum number of clusters to test when finding optimal strata.
    :return: Balanced DataFrame with undersampled majority class.
    """
    

    # Step 1: Filter majority class samples for MI calculation and stratification
    majority_class = df[df['y'] == 0].copy()

    # Step 2: Compute pairwise MI between majority class samples
    X_majority = majority_class.drop(columns=['y']).values  # Features of majority class
    N = len(X_majority)

    mi_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            mi = mutual_info_score(X_majority[i], X_majority[j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # Symmetric matrix

    print(mi_matrix)
    print(mi_matrix.shape) # 191 x 191 --> good

    
    # Step 3: Determine the number of strata
    if not num_strata_fixed:
        print("NON-FIXED strata")
        if use_features:
            # Use find_optimal_num_strata to determine the optimal number of clusters based on all features
            print("Optimal strata via all features (variant 1A)")
            # optimal_num_strata = find_optimal_num_strata(majority_class, use_features=True, max_clusters=max_clusters)
            optimal_num_strata = 2
            print(f"Optimal number of strata determined: {optimal_num_strata}")
        else:
            # Use find_optimal_num_strata to determine the optimal number of clusters based on MI
            print("Optimal strata via MI (variant 1B)")
            # optimal_num_strata = find_optimal_num_strata(majority_class, use_features=False, max_clusters=max_clusters)
            optimal_num_strata = 3
            print(f"Optimal number of strata determined: {optimal_num_strata}")
    else:
        # Use default fixed number of clusters
        print("FIXED strata")
        optimal_num_strata = 5
        print(f"Using fixed number of strata: {optimal_num_strata}")


    # Step 4: Group majority class samples into strata using clustering on MI matrix

    clustering = AgglomerativeClustering(n_clusters=optimal_num_strata, metric='precomputed', linkage='average')
    majority_class['stratum'] = clustering.fit_predict(1 - mi_matrix)  # Convert similarity to dissimilarity

    print(majority_class['stratum'].value_counts())

    # Step 5: Perform undersampling on majority class samples
    # Total number of samples to be sampled from the majority class (equal to the minority class size)
    minority_class_size = len(df[df['y'] == 1])

    # Group majority class by stratum
    strata_groups = majority_class.groupby('stratum')

    # Initial Equal Allocation: Sample the same number of samples from each stratum
    samples_per_stratum = minority_class_size // len(strata_groups)
    remaining_samples = minority_class_size - (samples_per_stratum * len(strata_groups))

    sampled_strata = []

    # Perform initial SRS on each stratum with equal allocation
    for stratum, group in strata_groups:
        num_samples_to_sample = min(samples_per_stratum, len(group))  # Ensure we don't sample more than available
        sampled_group = group.sample(n=num_samples_to_sample, random_state=42)
        sampled_strata.append(sampled_group)

    # Keep track of the number of samples we have so far
    sampled_majority_class = pd.concat(sampled_strata, axis=0)
    remaining_samples_needed = minority_class_size - len(sampled_majority_class)

    # If there are remaining samples to be drawn, perform SRS again
    while remaining_samples_needed > 0:
        print(f"Remaining samples needed: {remaining_samples_needed}")

        # Find strata that still have samples left to be drawn
        remaining_strata = majority_class.loc[~majority_class.index.isin(sampled_majority_class.index)]

        # Sample from these remaining strata using SRS
        strata_groups_remaining = remaining_strata.groupby('stratum')

        for stratum, group in strata_groups_remaining:
            # If there are still remaining samples in this stratum, sample them
            num_samples_to_sample = min(len(group), remaining_samples_needed)
            sampled_group = group.sample(n=num_samples_to_sample, random_state=42)
            sampled_majority_class = pd.concat([sampled_majority_class, sampled_group], axis=0)
            
            # Reduce remaining samples needed
            remaining_samples_needed -= num_samples_to_sample
            
            # Exit early if no more samples are needed
            if remaining_samples_needed <= 0:
                break

    # # After the loop, ensure sampled majority class matches the minority class size
    # print(f"Sampled majority class size: {len(sampled_majority_class)}")

    # Step 6: Combine undersampled majority class with the minority class
    # minority_class = df[df['y'] == 1]  # Keep all minority class samples
    # final_df = pd.concat([pd.concat(undersampled_majority_dfs), minority_class])
    minority_class = df[df['y'] == 1]  # Keep all minority class samples
    # print("minority df")
    # print(minority_class)

    # print("sampled majroity df")
    # print(sampled_majority_class)
    final_df = pd.concat([sampled_majority_class, minority_class])


    # Shuffle the resulting dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop unneeded columns
    final_df.drop(columns=['stratum'], inplace=True)

    print(final_df['y'].value_counts())

    return final_df


def sampling(df):
    """
    Undersample through maximizing mutual information: (3 parameters 12 VARIANTS)

    parameter 1: num_strata_fixed; is the number of strata fixed or not (2 values: True/ False)
    parameter 2: use_features; either use the data's features or mutual information to determine the number of strata IF num_strata_fixed is true (2 values: True/False)
    parameter 3: allocation; the allocation type use to sample data from each stratum (3 values: 'Neyman', 'Optimal', None)
    """
    output=sampling_through_mutual_information(df, num_strata_fixed=False, use_features=False, allocation=" ")

    return output
    
# NOTE: EXPERIMENT PART ENDS
#----------------------------------------------------------------------------------------------------------------------- 
#-----------------------------------------------------------------------------------------------------------------------


import time
# MAIN FUNCTION
def main():
    # Access to data
    data_path="https://raw.githubusercontent.com/Alex-Mak-MCW/Data_Sampling_Project/refs/heads/main/processed_breast_cancer_data.csv"


    # NOTE: TAKE OUT THE """ before and after each part to text
    
    #--------------------------------------------------------------------------------------------------
    """
    # BASELINE EXPERIMENT START
    start_time = time.time()

    # baseline testing
    baseline_testing(pd.read_csv(data_path))
    print(f"Baseline testing took {time.time() - start_time:.2f} seconds")
    # BASELINE EXPERIMENT ENDS
    """
    #--------------------------------------------------------------------------------------------------
    

    #--------------------------------------------------------------------------------------------------
    """
    # ACTUAL EXPERIMENT STARTS

    # Step 1: load the data
    start_time1 = time.time()
    data=pd.read_csv(data_path)
    print(data['y'].value_counts())
    print(f"Step 1: Loading data took {time.time() - start_time1:.2f} seconds")

    # Step 2: Sampling (ONLY DIFFERENCE)
    start_time2 = time.time()
    resampled_data=sampling(data)
    print(f"Step 2: Sampling took {time.time() - start_time2:.2f} seconds")
    
    # Step 3: do classification (and prediction)
    start_time3 = time.time()
    classification_and_evaluation(resampled_data)
    print(f"Step 3: Classification and Evaluation took {time.time() - start_time3:.2f} seconds")

    # ACTUAL EXPERIMENT ENDS
    """
    #--------------------------------------------------------------------------------------------------

main()
