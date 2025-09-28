### DFUSegNet: Boundary-aware hierarchical attentive fusion network with adaptive preprocessing for diabetic foot ulcer segmentation
A robust deep learning pipeline for the precise segmentation of Diabetic Foot Ulcers (DFUs) from clinical images. This project implements a sophisticated multi-scale, attention-guided fusion network to achieve state-of-the-art performance. Paper link: https://www.sciencedirect.com/science/article/pii/S0950705125013632  or https://shorturl.at/N2DVQ

## 🎯Highlights
- Multiscale Feature Fusion: Utilizes a novel approach with multiscale feature fusion in the encoder and a dual-mode attention mechanism in the decoder.
- Adaptive Preprocessing: Integrates a learnable image preprocessing module that adaptively enhances input quality and optimizes features for segmentation.
- Enhanced Feature Representation: Employs a boundary enhancer and multi-resolution positional attention to sharpen features and capture critical spatial context.
- State-of-the-Art Performance: Achieves superior performance and reliability in diabetic foot ulcer segmentation across multiple benchmark datasets.

## 📄 Abstract 
Diabetic Foot Ulcers (DFUs) are a severe complication of diabetes, often leading to lower limb amputation and increased patient morbidity. Accurate segmentation of DFUs is essential for effective wound assessment, treatment planning, and healing monitoring. This paper introduces a novel deep learning framework, DFUSegNet, for accurate segmentation of DFUs and other chronic wounds. The proposed architecture seamlessly integrates a learnable image preprocessor (LIP) to enhance input quality and a hierarchical encoder for capturing multiscale and multiresolution wound features. A boundary enhancer (BE) sharpens ulcer edges, while the multiresolution positional attention (MPA) module emphasizes critical spatial details. Extracted features by the encoder are refined through a local-global feature aggregation (LGFA) module before being processed by a dual-mode attention-guided hierarchical decoder, ensuring precise and robust segmentation. Extensive quantitative and qualitative evaluations on the DFUC, FUSeg, and AZH Wound datasets showcase the superior performance of DFUSegNet, achieving state-of-the-art IoU/F1-scores (in %) of 60.06/70.78 on DFUC, 79.06/85.76 on FUSeg, and 81.21/87.28 on AZH. Interpretability analysis further highlights the effectiveness of our MPA, BE modules, and dual-mode attention-guided decoder in progressively extracting intricate ulcer features. Despite encountering some anomalies in the datasets, DFUSegNet demonstrates immense potential for integration into knowledge-based systems within clinical workflows and telemedicine, enabling automated, high-precision DFU segmentation to support early diagnosis and effective wound management. While promising results validate its effectiveness, successful clinical deployment will require large, accurately annotated DFU datasets, laying the foundation for future advancements in automated DFU segmentation.

## 🚀 Getting StartedFollow these steps to set up the project environment and run the model.

Prerequisites
Python 3.9+pip and venv
Clone & Setup EnvironmentOpen your terminal, clone this repository, and set up the virtual environment.
1. Clone the repository
```git clone https://github.com/tushartalukder/DFUSegNet.git cd DFUSegNet```

2. Create and activate a virtual environment
```python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate```

3. Install all required dependencies
```pip install -r requirements.txt```

3. Configure PathsBefore running the model, you must configure your local data paths.
4. Open src/config.py with a text editor and update the variables to point to your dataset directories:# src/config.py

--- Data Paths ---
```BASE_PATH = "D:/path/to/your/WOUNDSEG/azh/split_1/" TEST_BASE_PATH = "D:/path/to/your/WOUNDSEG/azh/test/"```

--- Model Saving & Evaluation ---
```MODEL_SAVE_DIR = "D:/path/to/save/your/models/" EVAL_MODEL_PATH = "D:/path/to/your/trained_model.h5" # For evaluation```

### ⚙️ Usage
Once the setup is complete, you can train a new model or evaluate an existing one. 
### ▶️ Start Training
To start the training process from scratch, run the train.py script from the project's root directory.
```python -m src.train```
The console will display the training progress for each epoch. The best-performing model will be saved to the directory specified in your configuration. 

### ▶️ Evaluate a Model
To evaluate a pre-trained .h5 model, ensure the EVAL_MODEL_PATH in src/config.py is correctly set. Then, run the evaluation script:python -m src.evaluate 

This will load the test dataset and compute a comprehensive set of performance metrics.

### 📂 Project Structure
The project is organized into a clean and modular structure to separate concerns.
```wound-segmentation-project/
├── src/ 
│ ├── config.py # Main configuration for paths and hyperparameters 
│ ├── data_loader.py # Data loading, generators, and augmentation 
│ ├── losses.py # Custom loss function definitions 
│ ├── model_parts.py # Reusable model blocks (attention, conv blocks) 
│ ├── model.py # Main DFUSegNet model architecture 
│ ├── train.py # Main script to run the training process 
│ ├── evaluate.py # Script to evaluate a trained model 
│ └── utils.py # Helper functions (metrics, saving, etc.) 
├── requirements.txt # Project dependencies
└── README.md # This README file
```
## Cite as: 
@article{showrav2025dfusegnet, title={DFUSegNet: Boundary-aware hierarchical attentive fusion network with adaptive preprocessing for diabetic foot ulcer segmentation}, author={Showrav, Tushar Talukder and Hasan, Muhammad Zubair and Hasan, Md Kamrul}, journal={Knowledge-Based Systems}, pages={114323}, year={2025}, publisher={Elsevier} }
