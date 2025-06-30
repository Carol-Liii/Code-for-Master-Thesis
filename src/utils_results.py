import glob
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from collections import defaultdict
import matplotlib.pyplot as plt



################################################################################# binary ##################################################################################


# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def extract_method_from_model_name(model_name):
    """Extract ablation method from model name."""
    if 'mask' in model_name:
        return 'mask'
    elif 'remove' in model_name:
        return 'remove'
    return None

def extract_coarse_binary_feature_type(model_name):
    """Extract feature type from model name."""
    if 'sentiment_pos' in model_name:
        return 'sentiment_pos'
    elif 'sentiment_neg' in model_name:
        return 'sentiment_neg'
    elif 'function' in model_name:
        return 'function'
    elif 'hate' in model_name:
        return 'hate'
    return None

def load_coarse_binary_ablation_results(dataset_name, model_type='SVM'):
    """
    Load coarse binary ablation results for visualization.
    
    Parameters:
    -----------
    dataset_name : str
        'MAMI' or 'EXIST' or 'EXIST2024'
    model_type : str
        'SVM' or 'BERT' or 'BOTH'
        
    Returns:
    --------
    dict : Loaded results organized by feature type and method
    """
    print(f"🔍 Loading {model_type} coarse binary ablation results for {dataset_name}")
    
    # Determine which directories to search
    if model_type.upper() == 'BOTH':
        search_dirs = ["evaluation/results/binary/SVM", "evaluation/results/binary/BERT"]
    elif model_type.upper() == 'BERT':
        search_dirs = ["evaluation/results/binary/BERT"]
    else:  # Default to SVM
        search_dirs = ["evaluation/results/binary/SVM"]
    
    dataset_lower = dataset_name.lower()
    if dataset_name in ["EXIST2024", "EXIST"]:
        dataset_lower = "exist2024"  # Normalize naming
    
    results = {}
    total_files_found = 0
    
    for results_dir in search_dirs:
        if not os.path.exists(results_dir):
            print(f"⚠️  Directory not found: {results_dir}")
            continue
            
        print(f"   📁 Searching in: {results_dir}")
        
        # Pattern to match coarse binary results
        patterns = [
            f"{results_dir}/*{dataset_lower}*bin_results.json",
            f"{results_dir}/*{dataset_lower}*_results.json",
            f"{results_dir}/*{dataset_lower}*bert_results.json",  # For BERT results
        ]
        
        found_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            found_files.extend(files)
        
        # Remove duplicates
        found_files = list(set(found_files))
        total_files_found += len(found_files)
        print(f"   ✅ Found {len(found_files)} files in {results_dir}")
        
        for file_path in found_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                filename = os.path.basename(file_path)
                model_name = filename.replace('_results.json', '').replace('_bin_results.json', '').replace('_bert_results.json', '')
                
                # Skip baseline models
                if 'baseline' in model_name.lower():
                    continue
                    
                # Extract feature type and method from model name
                feature_type = extract_coarse_binary_feature_type(model_name)
                method = extract_method_from_model_name(model_name)
                
                if feature_type and method:
                    # Add model type info to distinguish SVM vs BERT
                    current_model_type = 'BERT' if 'bert' in results_dir.lower() else 'SVM'
                    
                    if feature_type not in results:
                        results[feature_type] = {}
                    
                    # Store with model type prefix to avoid conflicts
                    key = f"{method}_{current_model_type}"
                    results[feature_type][key] = data
                    results[feature_type][key]['model_type'] = current_model_type
                    
                    print(f"      ✅ Loaded: {feature_type} - {key}")
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    print(f"📊 Total: Found {total_files_found} files, loaded {len(results)} feature types")
    return results

def load_binary_baseline_results(dataset_name, model_type='SVM'):
    """
    Load baseline metrics for comparison with support for both SVM and BERT.
    """
    print(f"🔍 Loading {model_type} baseline results for {dataset_name}")
    
    # Determine which directories to search
    if model_type.upper() == 'BERT':
        search_dirs = ["evaluation/results/binary/BERT"]
        baseline_patterns = [
            f"evaluation/results/binary/BERT/*baseline*{dataset_name}*results.json",
            f"evaluation/results/binary/BERT/bert_baseline*{dataset_name.lower()}*results.json"
        ]
    else:  # SVM
        search_dirs = ["evaluation/results/binary/SVM"] 
        if dataset_name in ["EXIST2024", "EXIST"]:
            baseline_patterns = [
                "evaluation/results/binary/SVM/svm_baseline_bow_EXIST2024_bin_baseline_results.json",
                f"evaluation/results/binary/SVM/svm_baseline_bow_{dataset_name}_bin_baseline_results.json",
                f"evaluation/results/binary/SVM/svm_baseline_bow_EXIST_bin_baseline_results.json"
            ]
        else:  # MAMI
            baseline_patterns = [
                "evaluation/results/binary/SVM/svm_baseline_bow_MAMI_bin_baseline_results.json",
                f"evaluation/results/binary/SVM/svm_baseline_bow_{dataset_name}_bin_baseline_results.json",
                f"evaluation/results/binary/SVM/svm_baseline_bow_MAMI_baseline_results.json"
            ]
    
    # Try to find baseline files
    for baseline_file in baseline_patterns:
        # Also try glob pattern matching
        glob_files = glob.glob(baseline_file)
        all_baseline_files = [baseline_file] + glob_files
        
        for bf in all_baseline_files:
            if os.path.exists(bf):
                try:
                    with open(bf, 'r') as f:
                        baseline_data = json.load(f)
                    
                    print(f"✅ Loaded {model_type} baseline from: {bf}")
                    
                    # Method 1: Try to get from direct keys (new format)
                    if 'macro_f1' in baseline_data:
                        return {
                            'macro_f1': baseline_data.get('macro_f1', 0.7),
                            'macro_precision': baseline_data.get('precision_macro', 0.7),
                            'macro_recall': baseline_data.get('recall_macro', 0.7),
                            'model_type': model_type
                        }
                    
                    # Method 2: Try to get from per_label_metrics (sklearn format)
                    elif 'per_label_metrics' in baseline_data:
                        per_label_metrics = baseline_data.get('per_label_metrics', {})
                        macro_avg = per_label_metrics.get('macro avg', {})
                        
                        return {
                            'macro_f1': macro_avg.get('f1-score', 0.7),
                            'macro_precision': macro_avg.get('precision', 0.7),
                            'macro_recall': macro_avg.get('recall', 0.7),
                            'model_type': model_type
                        }
                        
                except Exception as e:
                    print(f"❌ Error loading baseline from {bf}: {e}")
                    continue
    
    # If no baseline file found, use default values based on model type and dataset
    print(f"⚠️ No {model_type} baseline file found for {dataset_name}, using default values")
    
    if model_type.upper() == 'BERT':
        if dataset_name == "MAMI":
            return {
                'macro_f1': 0.709,      
                'macro_precision': 0.739,
                'macro_recall': 0.716,
                'model_type': 'BERT'
            }
        else:  # EXIST2024
            return {
                'macro_f1': 0.657,      
                'macro_precision': 0.657,
                'macro_recall': 0.658,
                'model_type': 'BERT'
            }
    else:  # SVM
        if dataset_name == "MAMI":
            return {
                'macro_f1': 0.660,      
                'macro_precision': 0.710,
                'macro_recall': 0.674,
                'model_type': 'SVM'
            }
        else:  # EXIST2024
            return {
                'macro_f1': 0.640,      
                'macro_precision': 0.644,
                'macro_recall': 0.639,
                'model_type': 'SVM'
            }





############################################################################### multilabel ################################################################################


def extract_multilabel_category_from_model_name(model_name):
    """
    Extract fine-grained category from model name.
    Expected formats:
    - MAMI_svm_ablation_neg_sadness_remove_hierarchy -> 'emotion_sadness'
    - MAMI_svm_ablation_func_pronouns_mask_hierarchy -> 'function_pronouns'
    - MAMI_svm_ablation_hate_ps_remove_hierarchy -> 'hate_ps'
    """
    try:
        # Handle POS ablation models (different naming pattern)
        if 'pos_ablation' in model_name:
            parts = model_name.split('_')
            if 'pos' in parts and 'ablation' in parts:
                ablation_index = parts.index('ablation')
                if ablation_index + 1 < len(parts):
                    pos_category = parts[ablation_index + 1]
                    return f"pos_{pos_category.lower()}"
                    
        # Remove dataset prefix and common parts
        parts = model_name.split('_')
        
        # Find the ablation part
        if 'ablation' in parts:
            ablation_index = parts.index('ablation')
            
            # Extract the parts after 'ablation'
            remaining_parts = parts[ablation_index + 1:]
            
            if len(remaining_parts) >= 2:
                category_type = remaining_parts[0] # neg, func, hate, etc.
                category_detail = remaining_parts[1] # sadness, pronouns, ps, etc.
                
                # Handle different category formats
                if category_type == 'neg':
                    # For negative emotions: neg_sadness -> emotion_sadness
                    return f"emotion_{category_detail}"
                    
                elif category_type == 'func':
                    # For function words: func_pronouns -> function_pronouns  
                    return f"function_{category_detail}"
                    
                elif category_type == 'hate': 
                    # For hate speech: hate_ps -> hate_ps
                    return f"hate_{category_detail}"
                    
                elif category_type == 'sentiment':
                    # For sentiment: sentiment_pos -> sentiment_positive
                    if category_detail == 'pos':
                        return 'sentiment_positive'
                    elif category_detail == 'neg':
                        return 'sentiment_negative'
                    else:
                        return f"sentiment_{category_detail}"
                        
                else:
                    # Generic case: just combine with underscore
                    return f"{category_type}_{category_detail}"
            
            elif len(remaining_parts) == 1:
                # Single category like 'function', 'hate'
                category = remaining_parts[0]
                
                # Handle special cases
                if category == 'all':
                    # Skip combined categories like 'all_neg_emotions'
                    if len(remaining_parts) >= 3 and remaining_parts[1] == 'neg':
                        return None  # Skip these
                        
                return category
        
        # Fallback patterns for different naming conventions
        fallback_patterns = [
            ('sentiment_pos', 'sentiment_positive'),
            ('sentiment_neg', 'sentiment_negative'), 
            ('function_words', 'function_general'),
            ('hate_speech', 'hate_general')
        ]
        
        for pattern, replacement in fallback_patterns:
            if pattern in model_name:
                return replacement
                
        return None
        
    except Exception as e:
        print(f"Error extracting category from {model_name}: {e}")
        return None


def load_multilabel_baseline_results(baseline_model_name, dataset_name):
    """
    Load baseline results from saved JSON files with updated paths.
    """
    # Try different possible baseline file patterns with subdirectories
    baseline_path = [
        f"evaluation/results/multi-label/SVM/svm_baseline_bow_hierarchy_{dataset_name}_results.json"
    ]
    
    for baseline_file in baseline_path:
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                
                # Try different ways to extract macro F1
                macro_f1 = None
                
                # Method 1: Direct key access
                if 'macro_f1' in baseline_data:
                    macro_f1 = baseline_data['macro_f1']
                    
                # Method 2: From per_label_metrics
                elif 'per_label_metrics' in baseline_data and 'macro avg' in baseline_data['per_label_metrics']:
                    macro_f1 = baseline_data['per_label_metrics']['macro avg']['f1-score']
                
                if macro_f1 is not None:
                    print(f"✅ Loaded baseline from: {baseline_file}")
                    
                    return {
                        'binary_f1': baseline_data.get('binary_f1', 0),
                        'multilabel_f1': baseline_data.get('multilabel_f1', 0),
                        'macro_f1': macro_f1,  
                        'per_label_f1': {
                            label: baseline_data['per_label_metrics'][label]['f1-score']
                            for label in baseline_data.get('per_label_metrics', {}).keys()
                            if label not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg']
                        } if 'per_label_metrics' in baseline_data else {}
                    }
            except Exception as e:
                print(f"❌ Error loading baseline results from {baseline_file}: {e}")
                continue
    
    return None

def load_multilabel_model_results(file_path):
    """
    Load model evaluation results from a specific file path.
    """
    if not os.path.exists(file_path):
        print(f"❌ Results file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        
        # Try different ways to extract macro F1
        macro_f1 = None
        
        # Method 1: Direct key access
        if 'macro_f1' in model_data:
            macro_f1 = model_data['macro_f1']
        # Method 2: From per_label_metrics
        elif 'per_label_metrics' in model_data and 'macro avg' in model_data['per_label_metrics']:
            macro_f1 = model_data['per_label_metrics']['macro avg']['f1-score']
        
        if macro_f1 is not None:
            return {
                'multilabel_f1': model_data.get('multilabel_f1', 0),
                'macro_f1': macro_f1,  
                'per_label_f1': {
                    label: model_data['per_label_metrics'][label]['f1-score']
                    for label in model_data.get('per_label_metrics', {}).keys()
                    if label not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg']
                } if 'per_label_metrics' in model_data else {}
            }
        else:
            print(f"❌ Could not extract macro F1 from {file_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model results from {file_path}: {e}")
        return None

def collect_multilabel_ablation_results(dataset_name):
    """
    Collect all ablation results for the dataset.
    Search in subdirectories: binary/, multi-label/, pos/
    """
    all_results = {}
    
    # Search in all subdirectories
    subdirs = ["multi-label/SVM"]  
    
    # Map dataset names to actual file prefixes
    if dataset_name in ["EXIST2024", "EXIST"]:
        search_patterns = [
            "*exist*_results.json",
            "*exist2024*_results.json",
            "EXIST2024_*_results.json",
            "svm_pos_ablation_*exist*_results.json"
        ]
        search_name = "EXIST2024"
        
    elif dataset_name == "MAMI":
        search_patterns = [
            "*mami*_results.json",
            "MAMI_*_results.json", 
            "svm_pos_ablation_*mami*_results.json"
        ]
        search_name = "MAMI"
    else:
        search_patterns = [f"*{dataset_name}*_results.json"]
        search_name = dataset_name
    
    print(f"🔍 Searching for {search_name} ablation results...")
    
    found_files = []
    
    # Search in each subdirectory
    for subdir in subdirs:
        if subdir:
            results_dir = f"evaluation/results/{subdir}"
        else:
            results_dir = "evaluation/results"
            
        if not os.path.exists(results_dir):
            continue
            
        print(f"   📁 Searching in: {results_dir}")
        
        for pattern in search_patterns:
            full_pattern = os.path.join(results_dir, pattern)
            files = glob.glob(full_pattern)
            found_files.extend(files)
            if files:
                print(f"      ✅ Found {len(files)} files matching: {pattern}")
    
    # Remove duplicates
    found_files = list(set(found_files))
    print(f"📁 Total files found: {len(found_files)}")
    
    if not found_files:
        print("❌ No result files found!")
        return {}
    
    loaded_count = 0
    for file_path in found_files:
        file_name = os.path.basename(file_path)
        model_name = file_name.replace('_results.json', '')
        
        # Skip baseline models
        if 'baseline' in model_name.lower():
            continue
        
        # Load results using the file path
        results = load_multilabel_model_results(file_path)
        if results:
            all_results[model_name] = results
            loaded_count += 1
    
    print(f"✅ Successfully loaded {loaded_count} ablation results")
    return all_results




############################################################################ POS ABLATION FUNCTIONS #######################################################################

def extract_pos_category_from_model_name(model_name):
    """
    Extract POS category from model name for both SVM and BERT.
    Handles patterns like:
    - svm_pos_ablation_verb_mami -> pos_verb
    - bert_pos_ablation_clean_noun_mami -> pos_noun
    """
    try:
        # Handle POS ablation models
        if 'pos_ablation' in model_name:
            parts = model_name.split('_')
            
            # Find 'pos' and 'ablation' indices
            pos_index = -1
            ablation_index = -1
            
            for i, part in enumerate(parts):
                if part == 'pos':
                    pos_index = i
                elif part == 'ablation':
                    ablation_index = i
            
            if pos_index != -1 and ablation_index != -1 and ablation_index > pos_index:
                # Look for POS tag after 'ablation'
                start_search = ablation_index + 1
                
                # Skip 'clean' if present (for BERT cleaned models)
                if start_search < len(parts) and parts[start_search] == 'clean':
                    start_search += 1
                
                if start_search < len(parts):
                    pos_tag = parts[start_search]
                    
                    # Validate that it's a valid POS tag
                    valid_pos_tags = ['adj', 'adv', 'intj', 'noun', 'propn', 'verb']
                    if pos_tag.lower() in valid_pos_tags:
                        return f"pos_{pos_tag.lower()}"
        
        return None
        
    except Exception as e:
        print(f"Error extracting POS category from {model_name}: {e}")
        return None



def analyze_pos_ablation_universal(dataset_name, model_type='SVM'):
    """
    Universal POS ablation analysis that works for both SVM and BERT.
    """
    print(f"🔍 {model_type} POS ABLATION ANALYSIS (MACRO F1) - {dataset_name}")
    print("=" * 70)
    
    # Load baseline results
    baseline_results = load_binary_baseline_results(dataset_name, model_type)
    if not baseline_results:
        print(f"❌ Could not load {model_type} baseline results")
        return []
    
    print(f"✅ Loaded {model_type} baseline results - Macro F1: {baseline_results['macro_f1']:.3f}")
    
    # Determine POS results directory based on model type
    if model_type.upper() == 'BERT':
        pos_dir = "evaluation/results/POS/BERT"
    else:
        pos_dir = "evaluation/results/POS/SVM"
    
    if not os.path.exists(pos_dir):
        print(f"❌ POS results directory not found: {pos_dir}")
        return []
    
    print(f"📁 Searching in: {pos_dir}")
    
    # Search for POS files 
    if dataset_name in ["EXIST2024", "EXIST"]:
        patterns = [
            f"{pos_dir}/*exist*_results.json",
            f"{pos_dir}/*exist2024*_results.json",
            f"{pos_dir}/{model_type.lower()}_pos_ablation_*exist*_results.json",
            f"{pos_dir}/{model_type.lower()}_pos_ablation_clean_*exist*_results.json"  # For cleaned BERT
        ]
    else:  # MAMI
        patterns = [
            f"{pos_dir}/*mami*_results.json", 
            f"{pos_dir}/{model_type.lower()}_pos_ablation_*mami*_results.json",
            f"{pos_dir}/{model_type.lower()}_pos_ablation_clean_*mami*_results.json"  # For cleaned BERT
        ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
        print(f"   Found {len(files)} files matching: {os.path.basename(pattern)}")
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if not found_files:
        print("❌ No POS result files found!")
        return []
    
    print(f"📊 Total files found: {len(found_files)}")
    
    # Load POS results
    pos_results = {}
    for file_path in found_files:
        model_name = os.path.basename(file_path).replace('_results.json', '')
        results = load_multilabel_model_results(file_path)
        if results:
            pos_results[model_name] = results
            print(f"   ✅ Loaded: {model_name}")
    
    if not pos_results:
        print("❌ No valid POS results loaded!")
        return []
    
    # Calculate POS F1 drops using MACRO F1
    baseline_macro_f1 = baseline_results['macro_f1']
    pos_drops = []
    
    for model_name, results in pos_results.items():
        # Extract POS category from model name
        pos_category = extract_pos_category_from_model_name(model_name)
        
        if pos_category and pos_category.startswith('pos_'):
            f1_drop = baseline_macro_f1 - results['macro_f1']
            pos_drops.append({
                'category': pos_category,
                'f1_drop': f1_drop,
                'macro_f1': results['macro_f1'],
                'model_count': 1,  
                'model_name': model_name,
                'model_type': model_type
            })
    
    # Sort by F1 drop (descending)
    pos_drops.sort(key=lambda x: x['f1_drop'], reverse=True)
    
    # Display results
    print(f"\n📊 {model_type} BASELINE MACRO F1: {baseline_macro_f1:.3f}")
    print(f"\n{model_type} POS ABLATION - MACRO F1 DROPS")
    print("=" * 70)
    print("Rank Category                           F1 Drop  Macro F1   Model Type")
    print("-" * 70)
    
    for i, drop_info in enumerate(pos_drops, 1):
        print(f"{i:<4} {drop_info['category']:<35} {drop_info['f1_drop']:.3f}     {drop_info['macro_f1']:.3f}    {drop_info['model_type']}")
    
    return pos_drops



def create_pos_visualization_universal(dataset_name, model_type='SVM', save_plots=True):
    """
    Create POS ablation visualization for a single model type with custom labels.
    """
    print(f"🎨 Creating {model_type} POS ablation visualization for {dataset_name}")
    print("=" * 70)
    
    # Get POS ablation data
    pos_drops = analyze_pos_ablation_universal(dataset_name, model_type)
    
    if not pos_drops:
        print("❌ No POS data found for visualization")
        return None
    
    # Convert to DataFrame
    pos_df = pd.DataFrame(pos_drops)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. F1 Drop Bar Chart
    ax1 = axes[0]
    pos_categories = [drop['category'].replace('pos_', '').upper() for drop in pos_drops]
    f1_drops = [drop['f1_drop'] for drop in pos_drops]
    
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in f1_drops]
    bars = ax1.bar(range(len(pos_categories)), f1_drops, color=colors, alpha=0.8)
    
    ax1.set_title(f'{model_type} POS Ablation: F1 Drops', fontweight='bold', fontsize=13)
    ax1.set_xlabel('POS Category')
    ax1.set_ylabel('F1 Drop')
    ax1.set_xticks(range(len(pos_categories)))
    ax1.set_xticklabels(pos_categories, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add custom value labels
    for bar, value in zip(bars, f1_drops):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.001 if value >= 0 else -0.001),
                f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    # 2. Absolute F1 Scores
    ax2 = axes[1]
    macro_f1_scores = [drop['macro_f1'] for drop in pos_drops]
    
    bars = ax2.bar(range(len(pos_categories)), macro_f1_scores, alpha=0.8, color='#2ca02c')
    ax2.set_title(f'{model_type} POS Ablation: Absolute F1 Scores', fontweight='bold', fontsize=13)
    ax2.set_xlabel('POS Category')
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_xticks(range(len(pos_categories)))
    ax2.set_xticklabels(pos_categories, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add baseline line
    baseline_f1 = pos_drops[0]['f1_drop'] + pos_drops[0]['macro_f1']  # Calculate baseline
    ax2.axhline(y=baseline_f1, color='black', linestyle='--', alpha=0.7, 
                label=f'Baseline ({baseline_f1:.3f})', linewidth=2)
    ax2.legend()
    
    # Add custom value labels
    for bar, value in zip(bars, macro_f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. POS Impact Ranking (Horizontal)
    ax3 = axes[2]
    # Sort by F1 drop for ranking
    sorted_pos = sorted(pos_drops, key=lambda x: x['f1_drop'], reverse=True)
    sorted_categories = [drop['category'].replace('pos_', '').upper() for drop in sorted_pos]
    sorted_drops = [drop['f1_drop'] for drop in sorted_pos]
    
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in sorted_drops]
    bars = ax3.barh(range(len(sorted_categories)), sorted_drops, color=colors, alpha=0.7)
    
    # Add custom horizontal bar labels
    for i, value in enumerate(sorted_drops):
        label_x = value + (0.0005 if value >= 0 else -0.0005)
        ha = 'left' if value >= 0 else 'right'
        ax3.text(label_x, i, f'{value:.3f}', 
                ha=ha, va='center', fontweight='bold', fontsize=9)
    
    ax3.set_yticks(range(len(sorted_categories)))
    ax3.set_yticklabels(sorted_categories)
    ax3.set_xlabel('F1 Drop')
    ax3.set_title(f'{model_type} POS Impact Ranking', fontweight='bold', fontsize=13)
    ax3.grid(axis='x', alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = axes[3]
    
    # Calculate summary statistics
    avg_drop = np.mean(f1_drops)
    std_drop = np.std(f1_drops)
    max_drop = max(f1_drops)
    min_drop = min(f1_drops)
    avg_f1 = np.mean(macro_f1_scores)
    
    summary_text = f"{model_type} POS Ablation Summary:\n\n"
    summary_text += f"Average F1 Drop: {avg_drop:.3f}\n"
    summary_text += f"Std F1 Drop: {std_drop:.3f}\n"
    summary_text += f"Max F1 Drop: {max_drop:.3f}\n"
    summary_text += f"Min F1 Drop: {min_drop:.3f}\n"
    summary_text += f"Average F1: {avg_f1:.3f}\n"
    summary_text += f"Baseline F1: {baseline_f1:.3f}\n\n"
    summary_text += f"Most Impactful: {sorted_categories[0]}\n"
    summary_text += f"Least Impactful: {sorted_categories[-1]}"
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, 
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_title(f'{model_type} POS Summary', fontweight='bold', fontsize=13)
    ax4.axis('off')
    
    # Overall title and layout
    plt.suptitle(f'{dataset_name} {model_type} POS Ablation Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    # Save plot
    if save_plots:
        os.makedirs("evaluation/visualizations", exist_ok=True)
        filename = f"evaluation/visualizations/{dataset_name}_{model_type}_POS_ablation.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved POS visualization to: {filename}")
    
    plt.show()
    
    return pos_df