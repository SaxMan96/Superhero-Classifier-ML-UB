# Superhero-Classifier-ML-UB
Predicting the superhero alignment according to her traits and powers.

# Requirements

- Git - for versioning code, maybe we will try tasks or other features - [GIT](https://git-scm.com/downloads) - installation file
- Conda (or other virtual environment manager) - we will probably create our environment file - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) - installation file
  - python, numpy, pandas ... etc
  - kaggle, jupyter
- Kaggle account - to download and submit to Kaggle

# Setting up environment

- Clone this repository (you don't need to download the data from Kaggle, because it is already in repository, but try do do this by Kaggle API so you know that you will be able to submit it if needed)
- Create virtual environment from attached file (TO BE DONE)
- Run notebook (TO BE DONE)

# Project ML Stack

- Data analysis - [stats.py](https://github.com/SaxMan96/Superhero-Classifier-ML-UB/blob/master/stats.py)
  - Nans
  - Unique
  - Value Types
  - Min/Max
- Data engineering - [utilities.py](https://github.com/SaxMan96/Superhero-Classifier-ML-UB/blob/master/utilities.py)
  - NaNs
  
    - Filling With 0
  - Manual Feature Manipulation
    - (TO BE DONE)
- Feature Encoding
  - OrdinalEncoder
  - BinaryEncoder
  - OneHotEncoder
  - BaseNEncoder
  - HashingEncoder
- Feature selection
  - (TO BE DONE)
- Dimensionality Reduction
  - PCA
- Unbalanced Data Manipulations
  - SMOTE - Up-sampling
- Models
  - DecisionTreeClassifier
  - RandomForestClassifier
  - KNeighborsClassifier
  - LogisticRegression
  - XGBClassifier
  - SVC
- Pipeline
  - StandardScaler + Classifier
- Parameter optimalization
  - GridSearch + CV
- Ensamble
  - Voting Classifier
