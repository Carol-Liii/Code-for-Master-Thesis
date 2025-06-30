# Overviw
This project investigates the role of different linguistic features in automated detection of misogynistic and sexist content through systematic ablation studies using both traditional machine learning (SVM) and deep learning (BERT) approaches.

# Project structure
<pre>
thesis-project/
├── data/
│   ├── evaluation         # Gold Labels 
│   ├── datasets/          # Training, dev, test data for MAMI and EXIST2024
│   └── lexicons/          # Required linguistic resources
│       ├── NRC-Emotion-Lexicon/
│       └── hurtlex-master/
├── evaluation/
│   ├── predictions/       # Model predictions in various formats
│   ├── results/          # Evaluation metrics and analysis
│   │   ├── binary/SVM/
│   │   ├── multi-label/SVM/
│   │   └── POS/SVM/
│   └── exported_results/ # CSV exports for analysis
├── src/
│   ├── BERT main.ipynb  # Main experimental notebook for BERT model (run in Google Colab)
│   ├── SVM main.ipynb    # Main experimental notebook for SVM model
│   ├── utils_baseline.py # Core classification and evaluation functions
│   └── utils_experiments.py # Feature extraction and ablation utilities
├── LICENSE
├── README.md
└── requirements.txt
</pre>

# Thesis report
Full thesis document available at: [https://drive.google.com/file/d/1wsrNe_vciz8ZS-FtIcRFzDnMjQZHMHuv/view?usp=share_link
](https://www.overleaf.com/read/jvcprrsffdmj#2f2c2f)

# Data
Due to privacy reasons, dataset files are not included in this repository due to privacy restrictions. To obtain the dataset, please contact the original dataset creators.

Dataset:
MAMI (Multimodal Abuse and Misogyny Identification)
EXIST2024 (sEXism Identification in Social neTworks task at CLEF 2024)

## Required Lexicons

This project requires the following lexicons to be downloaded separately:

### NRC Emotion Lexicon
- **Source**: [http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- **File**: `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`
- **Place in**: `lexicons/NRC-Emotion-Lexicon/`

### HurtLex
- **Source**: [https://github.com/valeriobasile/hurtlex](https://github.com/valeriobasile/hurtlex)
- **File**: `hurtlex_EN.tsv`
- **Place in**: `lexicons/hurtlex-master/lexica/EN/1.2/`


# README
For SVM Experiments (Local)
Prerequisites:

bashpip install -r requirements.txt
python -m spacy download en_core_web_sm


1. Configure Paths: Update dataset and lexicon paths in SVM main.ipynb
2. Download Lexicons: Place NRC and HurtLex files in specified directories
3. Run Experiments: Execute all cells in SVM main.ipynb sequentially
4. Generate Output: Results will be saved in the evaluation directory
<pre>
evaluation/
├── predictions/           # Model predictions for evaluation
│   ├── MAMI/
│   └── EXIST2024/
├── results/              # Detailed JSON results
│   ├── binary/
│   │   └── SVM/          # SVM binary classification results
│   ├── multi-label/
│   │   └── SVM/          # SVM multi-label results  
│   └── POS/
│       └── SVM/          # SVM POS ablation results
└── exported_results/     # CSV files for analysis
</pre>

For BERT Experiments (Google Colab)
The BERT notebook is designed to run in Google Colab with GPU acceleration. Required packages will be installed within the notebook.
1. Setup Environment:

- Open BERT main.ipynb in Google Colab
- Mount Google Drive and set base path for your thesis materials
- Enable GPU runtime: Runtime → Change runtime type → Hardware accelerator: GPU

2. Configure Paths: Update dataset and lexicon paths in the notebook
3. Install Dependencies: Run the setup cells to install required packages
4. Execute Experiments: Run all cells sequentially for complete ablation study
