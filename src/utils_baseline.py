from typing import Dict, List, Any
import pandas as pd
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Import for PyEvALL evaluation library
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

### Evaluation metrics
def check_matrix(confusion_matrix, gold_labels, predicted_labels):
    """
    Check and adjust confusion matrix dimensions for proper evaluation.
    
    Parameters:
    - confusion_matrix (numpy.ndarray): The confusion matrix to check and potentially adjust
    - gold_labels (list or array): The gold standard (true) labels
    - predicted_labels (list or array): The predicted labels
    
    Returns:
    - numpy.ndarray: Properly dimensioned confusion matrix
    """
    if confusion_matrix.size == 1:
        tmp = confusion_matrix[0][0]
        confusion_matrix = np.zeros((2, 2))
        if (predicted_labels[1] == 0):
            # true negative
            if gold_labels[1] == 0:  
                confusion_matrix[0][0] = tmp
            # false negative
            else:  
                confusion_matrix[1][0] = tmp
        else:
            # false positive
            if gold_labels[1] == 0:  
                confusion_matrix[0][1] = tmp
            # true positive
            else:  
                confusion_matrix[1][1] = tmp
    return confusion_matrix

def compute_f1(predicted_values, gold_values):
    """
    Compute F1 score based on predicted and gold values.
    
    Parameters:
    - predicted_values (list or array): Predicted label values
    - gold_values (list or array): Gold standard (true) label values
    
    Returns:
    - float: Macro-averaged F1 score (average of positive and negative class F1 scores)
    """
    matrix = metrics.confusion_matrix(gold_values, predicted_values)
    matrix = check_matrix(matrix, gold_values, predicted_values)

    # Calculate precision and recall for positive label
    if matrix[0][0] == 0:
        pos_precision = 0.0
        pos_recall = 0.0
    else:
        pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

    # Calculate F1 for positive label
    if (pos_precision + pos_recall) != 0:
        pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    else:
        pos_F1 = 0.0

    # Calculate precision and recall for negative label
    neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    if neg_matrix[0][0] == 0:
        neg_precision = 0.0
        neg_recall = 0.0
    else:
        neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
        neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

    # Calculate F1 for negative label
    if (neg_precision + neg_recall) != 0:
        neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    else:
        neg_F1 = 0.0

    # Return macro-averaged F1 (average of positive and negative F1 scores)
    f1 = (pos_F1 + neg_F1) / 2
    return f1

def retrieve_label_values(ground_truth, model_submission, field_index):
    """
    Extract specific field values from ground truth and submission dictionaries.
    
    Parameters:
    - ground_truth (dict): Dictionary containing ground truth values
    - model_submission (dict): Dictionary containing model submission values
    - field_index (int): Index of the field to extract
    
    Returns:
    - tuple: Lists of extracted ground truth and model submission values
    """
    gold = []
    pred = []
    
    for k, v in ground_truth.items():
        gold.append(v[field_index])
        pred.append(model_submission[k][field_index])
        
    return gold, pred

def compute_binary_f1(ground_truth, model_submission):
    """
    Compute F1 score for binary classification task.
    
    Parameters:
    - ground_truth (dict): Dictionary containing ground truth values
    - model_submission (dict): Dictionary containing model submission values
    
    Returns:
    - float: F1 score for binary classification
    """
    gold, pred = retrieve_label_values(ground_truth, model_submission, 0)
    score = compute_f1(pred, gold)
    return score

def compute_multilabel_f1(truth_data, prediction_data, label_count):
    """
    Compute weighted F1 score for multi-label classification task.
    
    Parameters:
    - truth_data (dict): Dictionary containing ground truth values
    - prediction_data (dict): Dictionary containing model prediction values
    - label_count (int): Total number of labels including binary classification label
    
    Returns:
    - float: Weighted F1 score for multi-label classification
    """
    score_components = []
    occurrence_sum = 0
    
    # Skip first column (index 0) which contains binary classification labels
    for label_idx in range(1, label_count):
        true_values, predicted_values = retrieve_label_values(truth_data, prediction_data, label_idx)
        class_f1 = compute_f1(predicted_values, true_values)
        class_weight = true_values.count(True)
        occurrence_sum += class_weight
        score_components.append(class_f1 * class_weight)
    
    # Return weighted average, handling zero division case
    return sum(score_components) / occurrence_sum if occurrence_sum != 0 else 0.0

def load_data(filepath):
    """
    Load data from a tab-separated file and convert labels to boolean values.
    
    Parameters:
    - filepath (str): Path to the tab-separated data file
    
    Returns:
    - dict: Dictionary where keys are the first column values, and values are lists of boolean labels
    
    Raises:
    - ValueError: If file has inconsistent or incorrect format
    """
    result_dict = {}
    expected_columns = None
    
    with open(filepath) as input_file:
        csv_reader = csv.reader(input_file, delimiter='\t')
        line_num = 1
        
        for entry in csv_reader:
            if len(entry) < 2:  # ensure at least one label column is present
                raise ValueError(f'Wrong number of columns in line {line_num}, expected at least 2.')
            
            if expected_columns and len(entry) != expected_columns:
                raise ValueError(f'Inconsistent number of columns in line {line_num}.')
            
            expected_columns = len(entry)
            result_dict[entry[0]] = [bool(float(val)) for val in entry[1:]]
            line_num += 1
            
    return result_dict

def evaluate_f1_scores(gold_label_path, prediction_path, num_labels):
    """
    Evaluate scores for binary and multi-label classification tasks.
    
    Parameters:
    - gold_label_path (str): Path to the file containing gold standard labels
    - prediction_path (str): Path to the file containing model predictions
    - num_labels (int): Total number of labels (2 for binary classification, >2 for multi-label)
    
    Returns:
    - float or tuple: Binary score if num_labels=2, otherwise (binary_score, multilabel_score) tuple
    
    Raises:
    - ValueError: If submission is missing required keys
    """
    
    truth = load_data(gold_label_path)
    submission = load_data(prediction_path)
    
    # Ensure submission contains all necessary keys
    for key in truth.keys():
        if key not in submission:
            raise ValueError(f'Missing element {key} in submission')
    
    # Compute F1 metric for binary classification
    if num_labels == 2:
        binary_score = compute_binary_f1(truth, submission)
        return binary_score
    
    # Compute F1 for both binary classification and multi-label classification
    if num_labels > 2:
        binary_score = compute_binary_f1(truth, submission)
        multilabel_score = compute_multilabel_f1(truth, submission, num_labels)
        return binary_score, multilabel_score

### Binary classification 

def build_bin_classifier(X_train, y_train):
    """
    Create and train an SVM classifier with TF-IDF features.
    
    Parameters:
    - X_train (list): List of text documents for training
    - y_train (list): Binary labels for training documents
    
    Returns:
    - tuple: (svm_model, vectorizer) - Trained model and fitted vectorizer
    """
    # Create TF-IDF features
    vectorizer = TfidfVectorizer() # min_df=5,
    X_train_features = vectorizer.fit_transform(X_train)
    
    # Train SVM model
    svm_model = LinearSVC(max_iter=10000)
    svm_model.fit(X_train_features, y_train)
    
    return svm_model, vectorizer

def classify_data(X_test, model, vectorizer):
    """
    Classify test data using the trained model.
    
    Parameters:
    - X_test (list): List of text documents to classify
    - model: Trained classifier model
    - vectorizer: Fitted vectorizer for feature extraction
    
    Returns:
    - numpy.ndarray: Predicted labels for test documents
    """
    # Transform test data
    X_test_features = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_features)
    
    return y_pred


### Binary evaluation

def evaluate_binary_classification(gold_label_json, predictions_json, 
                                   y_true, y_pred, 
                                   gold_labels_txt, predictions_txt,
                                   label_names, 
                                   model_name="Model"):
    """
    Generate and print comprehensive evaluation metrics for binary classification.
    
    Parameters:
    - gold_label_json (str): Path to the file with gold labels in PyEvALL format
    - predictions_json (str): Path to the file with predicted labels in PyEvALL format
    - y_true (list or array): Gold labels
    - y_pred (list or array): Predicted labels
    - gold_labels_txt (str): Path to the file with gold labels in txt format for f1 metric
    - predictions_txt (str): Path to the file with predictions in txt format for f1 metric
    - label_names (list of str): The list of label names corresponding to the binary problem
    - model_name (str): Name of the model (default: "Model")

    Returns:
    - None: Results are printed to console and displayed as plots
    """
    
    y_pred = y_pred.tolist()
    
    # Print classification report with precision, recall, f1, and support metrics
    report = classification_report(y_true, y_pred, target_names=label_names, digits=3)
    print(f"{'-'*100}\nClassification Report for {model_name}:\n{report}\n{'-'*100}")
    
    # Generate confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(f"{'-'*100}\nConfusion matrix for {model_name}:")
    
    # # Print confusion matrix with 3 decimal places (normalized)
    # cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    # print("Normalized Confusion Matrix (3 decimal places):")
    # for i, row in enumerate(cf_matrix_normalized):
    #     formatted_row = [f"{val:.3f}" for val in row]
    #     print(f"{label_names[i]:>15}: {formatted_row}")
    # print()
    
    # Print raw confusion matrix
    print("Raw Confusion Matrix:")
    for i, row in enumerate(cf_matrix):
        print(f"{label_names[i]:>15}: {row.tolist()}")
    print()
    
    # Display confusion matrix as a heatmap with custom formatting
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=label_names)
    
    # Create figure with custom formatting for 3 decimal places
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with custom text formatting
    disp.plot(cmap=plt.cm.Greens, ax=ax, values_format='.3f' if np.any(cf_matrix < 1) else 'd')
    
    # If you want to show both raw counts and percentages with 3 decimals
    # Calculate percentages
    cf_matrix_percent = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()
    print(f"{'-'*100}")

    # PyEvALL evaluation metrics
    print(f"{'-'*100}\nPyEvaLL Metrics for {model_name}:\n")
    evaluator = PyEvALLEvaluation() 
    evaluation_params = dict() 
    evaluation_params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME
    metric_list = ["ICM", "ICMNorm", "FMeasure"] 
    evaluation_report = evaluator.evaluate(predictions_json, gold_label_json, metric_list, **evaluation_params) 
    evaluation_report.print_report()
    print(f"{'-'*100}")
    
    # MAMI F1 metric (macro-f1 for binary classification)
    print(f"{'-'*100}\n F1 Metrics for {model_name}:\n")
    n_labels = 2
    
    # Compute binary classification macro-F1 score
    score_bin = evaluate_f1_scores(gold_labels_txt, predictions_txt, n_labels)
    print(f"Binary classification macro-F1 score: {score_bin:.3f}")
    print(f"{'-'*100}")

    # Get structured classification report
    class_report_dict = classification_report(y_true, y_pred, 
                                            target_names=label_names, 
                                            zero_division=0,
                                            digits=3,
                                            output_dict=True)
    
    # Extract confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate binary F1 score
    score_bin = evaluate_f1_scores(gold_labels_txt, predictions_txt, n_labels)
    
    # Return structured results
    results = {
        'binary_f1': score_bin,
        'macro_f1': class_report_dict['macro avg']['f1-score'],
        'per_label_metrics': class_report_dict,
        'confusion_matrix': cf_matrix,
        'label_names': label_names
    }
    
    return results







### Multi-label classification

def build_multilabel_classifier(X_train, y_train, transform_strategy):
    """
    Create and train a multi-label text classification model using SVM with the specified strategy.
    
    Parameters:
    - X_train (list of str): Training text data
    - y_train (list): Multi-label training labels
    - transform_strategy (skmultilearn model wrapper): Multi-label classification strategy (BinaryRelevance/LabelPowerset)
    
    Returns:
    - tuple: (ml_model, vectorizer) - Trained multi-label model and fitted vectorizer
    """
    
    # Create TF-IDF vectorizer with NLTK tokenization
    vectorizer = TfidfVectorizer() # min_df=5
    X_train_features = vectorizer.fit_transform(X_train)
    
    # Configure multi-label model with LinearSVC base classifier
    ml_model = transform_strategy(LinearSVC(max_iter=10000))
    # Train the model
    ml_model.fit(X_train_features, y_train) 
    
    return ml_model, vectorizer




def build_hierarchical_multilabel_classifier(all_training_texts, binary_labels, 
                                             positive_subset_texts,category_labels, 
                                             test_texts, binary_label_name,
                                             category_label_names, strategy_class=BinaryRelevance):
    """
    Train a hierarchical classification model with two stages: binary and multi-label classification.
    
    This approach first classifies instances as positive/negative (e.g., misogynous/non-misogynous),
    then applies fine-grained classification only to positive instances.
    
    Parameters:
    - all_training_texts (list): All training text instances
    - binary_labels (list): Binary labels for all training instances
    - positive_subset_texts (list): Text instances with positive binary labels only
    - category_labels (list): Fine-grained category labels for positive instances
    - test_texts (list): Texts for evaluation
    - binary_label_name (str): Name of the binary label column
    - category_label_names (list): Names of fine-grained category labels
    - strategy_class: Multi-label classification strategy (default: BinaryRelevance)
    
    Returns:
    - tuple: (predictions_df, bin_model, bin_vec, ml_model, ml_vec)
            where:
            - predictions_df: DataFrame containing predictions for both binary and fine-grained labels
            - bin_model: Trained binary classification model
            - bin_vec: Text vectorizer for binary classification
            - ml_model: Trained multi-label classification model (None if no positive instances)
            - ml_vec: Text vectorizer for multi-label classification (None if no positive instances)

    """

    # First build binary model to predict positive instances (misogynous/sexist)
    bin_model, bin_vec  = build_bin_classifier(all_training_texts, binary_labels)
    binary_predictions = classify_data(test_texts, bin_model, bin_vec )

    # Filter positive instances （misogynous/sexist） for fine-grained classification
    positive_test_texts = pd.DataFrame(test_texts)[binary_predictions == 1][0].tolist()

    # Initialize predictions DataFrame with binary labels
    # Default all fine-grained labels to 0
    pred_df = pd.DataFrame({binary_label_name: binary_predictions})
    pred_df[category_label_names] = 0

    # Build multi-label model for fine-grained classification if there are positive instances
    if len(positive_test_texts) > 0:
        ml_model, ml_vec = build_multilabel_classifier(
            positive_subset_texts, category_labels, strategy_class)
        
        # Apply fine-grained classification only to positive instances
        multilabel_predictions = classify_data(
            positive_test_texts, ml_model, ml_vec)
        
        # Add fine-grained labels to positive instances in the predictions DataFrame
        pred_df.loc[binary_predictions == 1, category_label_names] = multilabel_predictions.toarray()
    else:
        # If no positive instances, create empty models (to maintain return structure)
        ml_model, ml_vec = None, None
    
    return pred_df, bin_model, bin_vec , ml_model, ml_vec



### Multi-label evaluation

def evaluate_multilabel_classification(gold_label_json, predictions_json, 
                                       y_true, y_pred, 
                                       gold_labels_txt, predictions_txt,
                                       label_names,
                                       hierarchy=True):
    """
    Evaluate the performance of a multi-label classification model with comprehensive metrics.

    Parameters:
    - gold_label_json (str): Path to the file with gold labels in PyEvALL format
    - predictions_json (str): Path to the file with predicted labels in PyEvALL format
    - y_true (array-like): Gold binary labels (multi-label binary matrix) for the test set
    - y_pred (array-like): Predicted binary labels (multi-label binary matrix) for the test set
    - label_names (list of str): The list of label names corresponding to the multi-label problem
    - gold_labels_txt (str): Path to the file with gold labels in txt format for MAMI f1 metric
    - predictions_txt (str): Path to the file with predictions in txt format for MAMI f1 metric
    - hierarchy (bool): Whether the evaluation considers hierarchical evaluation (first binary 
                       classification and then multi-label). Default is True.

    Returns:
    - dict: Evaluation results including F1 scores and per-label metrics
    """

    
    # Convert inputs to numpy arrays if they aren't already
    y_pred_array = y_pred.toarray() if not isinstance(y_pred, np.ndarray) else y_pred
    y_true_array = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true

    # Process binary classification data (first column)
    binary_true = y_true_array[:, 0]
    binary_pred = y_pred_array[:, 0]
    
    # Create negative class representation matrices
    negative_true = np.zeros((len(y_true_array), 1))
    negative_pred = np.zeros((len(y_pred_array), 1))
    
    # Set values for negative class (inverse of binary classification)
    negative_true[:, 0] = (binary_true == 0) #~binary_true.astype(bool)
    negative_pred[:, 0] = (binary_pred == 0) #~binary_pred.astype(bool)
    
    # Get multi-label classification data (all columns except first)
    multilabel_true = y_true_array[:, 1:]
    multilabel_pred = y_pred_array[:, 1:]
    
    # Create combined representation with negative class and multi-labels
    combined_true = np.hstack((negative_true, multilabel_true))
    combined_pred = np.hstack((negative_pred, multilabel_pred))
    
    # Update label names to include negative class
    updated_labels = [f"non-{label_names[0]}"] + label_names[1:]

    total_labels = len(updated_labels)
    
    # Print classification metrics with 3 decimal places
    print(f"{'-'*100}\nClassification Report:")
    class_report = classification_report(combined_true, combined_pred, 
                                        target_names=updated_labels, 
                                        zero_division=0,
                                        digits=3)
    print(f"{class_report}\n{'-'*100}")
    
    # GET STRUCTURED DATA: classification report as dictionary
    class_report_dict = classification_report(combined_true, combined_pred, 
                                            target_names=updated_labels, 
                                            zero_division=0,
                                            digits=3,
                                            output_dict=True)  # Return as dictionary
    
    # Generate and display confusion matrices (text only)
    print(f"{'-'*100}\nConfusion matrices:")
    confusion_matrices = multilabel_confusion_matrix(combined_true, combined_pred)

    
    
    # Print each confusion matrix with 3 decimal places
    for idx, matrix in enumerate(confusion_matrices):
        # Print numeric values with 3 decimal places for normalized matrix
        print(f"Confusion Matrix for '{updated_labels[idx]}' label:")
        print("Raw Matrix:")
        print(matrix)
        
        # Calculate and print normalized matrix with 3 decimal places
        if matrix.sum() > 0:
            matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized Matrix (3 decimal places):")
            for i, row in enumerate(matrix_normalized):
                formatted_row = [f"{val:.3f}" for val in row]
                print(f"  {['False', 'True'][i]:>5}: {formatted_row}")
        print()

    # Run PyEvALL evaluation
    print(f"{'-'*100}\nPyEvaLL Metrics:\n")
    evaluator = PyEvALLEvaluation() 
    evaluation_params = dict() 

    # Configure hierarchical evaluation if needed
    if hierarchy:
        label_hierarchy = {"yes": label_names, "no":[]} 
        evaluation_params[PyEvALLUtils.PARAM_HIERARCHY] = label_hierarchy
        metric_list = ["ICM", "ICMNorm", "FMeasure"] 
        
    else:
        metric_list = ["FMeasure"] 

  
    # Set report format
    evaluation_params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME
    
    # Run evaluation and print results
    evaluation_report = evaluator.evaluate(gold_label_json, predictions_json, metric_list, **evaluation_params)
    evaluation_report.print_report()
    print(f"{'-'*100}")


    # Calculate F1 metrics with 3 decimal places
    bin_score, ml_score = evaluate_f1_scores(gold_labels_txt, predictions_txt, total_labels)
    
    print(f"Binary classification macro-F1 score: {bin_score:.3f}")
    print(f"Multi-label classification weighted-F1 score: {ml_score:.3f}")
    print(f"{'-'*100}")
    
    # RETURN STRUCTURED RESULTS
    results = {
        'binary_f1': bin_score,
        'macro_f1': class_report_dict['macro avg']['f1-score'],
        'multilabel_f1': ml_score,
        'per_label_metrics': class_report_dict,
        'confusion_matrices': confusion_matrices,
        'label_names': updated_labels,
        'original_label_names': label_names
    }
    
    return results




### Convert predictions to PyEvALL format

def format_pred_for_pyevall(df, binary_label, labels, test_case, eval_type, pred_label):
    """
    Convert a DataFrame of labeled meme dataset into a format suitable for PyEvALLEvaluation.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing meme id and associated labels
    - binary_label (str): The name of the binary label column in the DataFrame (sexist or misogynous)
    - labels (list): A list of column names representing the labels for evaluation in the dataset
    - test_case (str): The test case identifier to be added as a new column, e.g. "MAMI" or "EXIST2024"
    - eval_type (str): The type of evaluation format to be used:
        - "binary": For binary classification
        - "hierarchical": For multi-label classification
    - pred_label (np.ndarray): A NumPy array containing predicted labels

    Returns:
    - list: A list of dictionaries formatted for PyEvALLEvaluation with structure:
        - test_case: The name of the dataset
        - id: The meme id
        - value: Labels in format appropriate for the evaluation type
    """

    # Convert files to input required by PyEvALLEvaluation
    pred_labels = df[["meme id"]].copy()
    
    # Add the test_case column as per the library requirements
    pred_labels.insert(0, "test_case", [test_case] * (len(pred_labels)), True) 
    
    if eval_type == "binary":
        
        # Format binary labels
        pred_labels["value"] = pred_label
        binary_labels = pred_labels
        
        # Convert values to "yes" and "no" as required by PyEvALL
        binary_labels = pred_labels.replace({"value":0}, "no").replace({"value":1}, "yes") 
        
        # Rename the id column to match requirements
        binary_labels.rename(columns={"meme id": "id"}, inplace=True) 
        
        # Convert "id" column to string values
        binary_labels["id"] = binary_labels["id"].astype(str)  
        labels_df = binary_labels
    
    elif eval_type == "hierarchical":
        
        # Format hierarchical multi-label data
        multilabel_labels = pred_labels[["test_case", "meme id"]].reset_index(drop=True)
        
        # Concatenate with dataset df along columns (axis=1) 
        multilabel_labels = pd.concat([multilabel_labels, pred_label], axis=1)
        
        # Extract only fine-grained category columns
        value_cols = multilabel_labels.columns[2:]
        
        # Create value column with list of labels where value is 1
        multilabel_labels["value"] = multilabel_labels[value_cols].apply(
            lambda row: [label for i, label in enumerate(labels[1:]) if row.iloc[i]] or ["no"], axis=1)
        multilabel_labels = multilabel_labels[["test_case", "meme id", "value"]]
        
        # Rename id column to match PyEvALL requirements
        multilabel_labels.rename(columns={"meme id": "id"}, inplace=True)
        
        # Convert "id" column to string values
        multilabel_labels["id"] = multilabel_labels["id"].astype(str)
        labels_df = multilabel_labels

    # Convert DataFrame to list of dictionaries as required by PyEvALL
    labels_list = labels_df.to_dict(orient="records") 

    return labels_list


### Convert predictions to MAMI f1 evaluation format

def format_pred_for_mami_f1(df, eval_type, pred_label):
    """
    Transform a DataFrame containing meme data into the format required by MAMI evaluation framework.
    
    Parameters:
    - df (pandas.DataFrame): DataFrame with meme identifiers and attribute data
    - eval_type (str): Specifies the evaluation approach:
        - "binary": Used for simple positive/negative classification
        - "hierarchical": Used for multi-level category classification
    - pred_label (np.ndarray or pd.DataFrame): Classification results from model prediction
    
    Returns:
    - pandas.DataFrame: Properly formatted data structure with IDs and classification values
    """
    # Extract just the identifier column to a new DataFrame
    pred_labels = df[["meme id"]].copy()
    
    if eval_type == "binary":
        # Binary case: simply attach prediction vector as a value column
        pred_labels["value"] = pred_label 
    
    else:
        # Handle multi-label scenario by first ensuring array format
        if not isinstance(pred_label, np.ndarray):
            pred_label = pred_label.toarray()
        # Transform array into structured DataFrame
        pred_df = pd.DataFrame(pred_label)
        # Normalize index sequencing for proper alignment
        pred_labels = pred_labels.reset_index(drop=True) 
        # Merge identifier column with prediction matrix
        pred_labels = pd.concat([pred_labels, pred_df], axis=1)  
        
    # Standardize identifier type to string format
    pred_labels["meme id"] = pred_labels["meme id"].astype(str) 
    
    return pred_labels



### Save predictions for evaluation

def write_labels_to_json(label_list, output_file, dataset_name, split_name, eval_type):
    """
    Write labels to JSON file for PyEvALL Evaluation.
    
    Parameters:
    - label_list (list): List of dictionaries containing test case, meme id and labels
    - output_file (str): Path to the output JSON file
    - dataset_name (str): Name of the dataset (e.g., MAMI, EXIST2024)
    - split_name (str): Name of the data split (e.g., training, test)
    - eval_type (str): Type of evaluation (binary, flat, hierarchical)
    
    Returns:
    - None
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)
    print(f"Saved {dataset_name} {split_name} split {eval_type} evaluation to {output_file}")


def write_labels_to_txt(labels_df, output_path, dataset_name, split_name):
    """
    Write labels to tab-separated text file for MAMI Evaluation.
    
    Parameters:
    - labels_df (pandas.DataFrame): DataFrame containing meme id and associated labels
    - output_path (str): Path to the output text file
    - dataset_name (str): Name of the dataset (e.g., MAMI, EXIST2024)
    - split_name (str): Name of the data split (e.g., training, test)
    
    Returns:
    - None
    """
    labels_df.to_csv(output_path, index=False, sep='\t', header=False)
    
    print(f"Saved {dataset_name} {split_name} split to {output_path}")


def save_evaluation(df, pred_dir, dataset_name, split_name, eval_type, model_name, predictions, binary_label, labels):
    """
    Store model evaluation data in structured format files and return their paths.
    
    Exports prediction results to both JSON (PyEvALL compatible) and TXT (MAMI F1 compatible) formats.
    
    Parameter:
    - df (pandas.DataFrame): Source dataset containing meme identifiers and ground truth labels
    - pred_dir (str): Target location for storing output files
    - dataset_name (str): Identifier for the evaluation corpus (e.g., "MAMI", "EXIST2024")
    - split_name (str): Partition identifier (e.g., "train", "dev", "test")
    - eval_type (str): Classification approach used ("binary", "hierarchical", "flat")
    - model_name (str): Classifier identifier for file naming
    - predictions (np.ndarray): Model output predictions matrix
    - binary_label (str): Primary category field name in the dataset
                         Used for yes/no categorization in certain evaluation types
    - labels (list): Field identifiers for all classification dimensions
    """

    dataset_name = "EXIST2024" if dataset_name == "EXIST" else dataset_name
    
    # Construct nested directory path for this specific dataset
    output_directory = os.path.join(pred_dir, dataset_name)
    
    # Ensure storage location exists
    os.makedirs(output_directory, exist_ok=True)

    # Generate PyEvALL-compatible representation
    prediction_records = format_pred_for_pyevall(
        df, binary_label, labels, dataset_name, eval_type, predictions
    )
    
    # Construct JSON output path and persist data
    json_filepath = os.path.join(
        output_directory, 
        f"{model_name}_{dataset_name}_{split_name}_{eval_type}.json"
    )
    write_labels_to_json(
        prediction_records, json_filepath, dataset_name, split_name, eval_type
    )

    # Handle DataFrame predictions by converting to numpy array
    prediction_data = predictions
    if isinstance(predictions, pd.DataFrame):
        prediction_data = predictions.to_numpy()
    
    # Generate MAMI F1 compatible representation
    mami_format_data = format_pred_for_mami_f1(df, eval_type, prediction_data)
    
    # Construct TXT output path and persist data
    txt_filepath = os.path.join(
        output_directory,
        f"{model_name}_{dataset_name}_{split_name}_answer.txt"
    )
    write_labels_to_txt(mami_format_data, txt_filepath, dataset_name, split_name)

    # Return both file paths for reference
    return json_filepath, txt_filepath

def save_predictions_csv(test_df, predictions, column_names, output_file):
    """
    Write predictions to a CSV file by adding new columns per predicted label(s).
    
    Parameters:
    - test_df (pd.DataFrame): DataFrame containing the original test instances and gold labels.
    - predictions (list or np.array): Model predictions.
    - column_names (list): A list of column names representing the labels for evaluation in the dataset.
    - output_file (str): Path to the output CSV file.
    """
    #convert predicted labels array to df with labels as column names + _prediction
    pred_df = pd.DataFrame(predictions, columns=[f"{col}_prediction" for col in column_names]) 
    
    #drop index to merge with predictions df
    test_df = test_df.reset_index(drop=True) 
    
    #save predictions while keeping original labels
    pred_df = pd.concat([test_df, pred_df], axis=1) 
    
    #save updated file
    pred_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")