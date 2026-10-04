# House Price Prediction: Machine Learning and Deep Learning

This project predicts residential house prices using the Ames Housing dataset. It contains two complementary experiments:

- **Classical machine learning:** several regression algorithms are trained and compared using scikit-learn-compatible APIs.
- **Deep learning:** a fully connected PyTorch neural network learns from numerical and categorical house features after a preprocessing pipeline.

The project is intended as a practical comparison of traditional regression models and a neural-network approach for tabular regression.

## Project Goals

The project answers four practical questions:

1. Which classical regression model gives the strongest house-price predictions?
2. How much can a model learn from a small, manually selected numerical feature set?
3. How does a neural network handle a mixture of numerical and categorical housing features?
4. How can model quality, cross-validation performance, and overfitting be tracked consistently?

## Dataset

The project uses the Ames Housing dataset.

- Kaggle source: [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- Local ML copy: `House Prices model/data/AmesHousing.csv`
- Target column: `SalePrice`
- The local Ames Housing CSV contains approximately 2,930 observations and 80 columns.

The dataset describes properties using information such as quality ratings, living area, garage capacity, basement size, year built, bathrooms, and sale price.

The two notebooks use related versions of the Ames data:

- The ML notebook reads the local CSV file.
- The DL notebook obtains an Ames dataset through OpenML with `fetch_openml(data_id=42165, as_frame=True, return_X_y=True)`.

Because the source copies and preprocessing pipelines are different, ML and DL metrics should be compared as indicative experiments rather than as a perfectly controlled benchmark.

## Repository Structure

```text
House price project/
├── House_prices_ML/
│   ├── app.ipynb
│   ├── README.md
│   └── House Prices model/
│       ├── code/
│       │   └── app.ipynb
│       ├── data/
│       │   └── AmesHousing.csv
│       └── info/
│           └── README.md
└── House_prices_DL/
	└── app.ipynb
```

The nested notebook at `House Prices model/code/app.ipynb` is the ML notebook copy that uses the local dataset path. The top-level notebooks are the main entry points for the two experiments.

## Machine Learning Workflow

The classical ML experiment follows this process:

1. Import pandas, NumPy, Matplotlib, seaborn, scikit-learn, XGBoost, LightGBM, CatBoost, and MLflow.
2. Load and inspect the Ames Housing data.
3. Remove identifier or highly incomplete columns such as `Order`, `PID`, and `Alley`.
4. Fill selected missing values with feature means, including `Lot Frontage`, `Garage Cars`, `Garage Area`, and `Total Bsmt SF`.
5. Select eight numerical predictors:

   - `Overall Qual`
   - `Gr Liv Area`
   - `Garage Cars`
   - `Garage Area`
   - `Total Bsmt SF`
   - `1st Flr SF`
   - `Full Bath`
   - `Year Built`

6. Split the data into training and test sets using an 80/20 split and `random_state=42`.
7. Train and evaluate multiple regression models.
8. Compare test performance with five-fold cross-validation and an overfitting gap.
9. Log metrics, parameters, and fitted models to MLflow.

### Classical Models

The notebook evaluates the following regressors:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- K-Neighbors Regressor
- Support Vector Regression is imported for experimentation but is not part of the final comparison in every notebook run.

### Classical Evaluation Metrics

The reusable `evaluate_model` function reports:

- **MSE:** mean squared error.
- **RMSE:** square root of MSE, expressed in the target's price units.
- **Train R²:** variance explained on the training data.
- **Test R²:** variance explained on unseen test data.
- **Cross-validation R²:** mean and standard deviation from five folds.
- **Overfitting gap:** the absolute difference between train and test R².

An overfitting warning is printed when the train/test R² gap is greater than `0.1`.

### Recorded Classical Results

The strongest recorded result in the notebook is CatBoost:

| Model | Test R² | RMSE | Five-fold CV R² |
| --- | ---: | ---: | ---: |
| CatBoost | 0.8928 | approximately 29,317 | 0.8742 ± 0.0189 |
| Gradient Boosting | 0.8820 | not consistently recorded in the summary | not consistently recorded in the summary |

These values depend on the exact notebook version, installed library versions, and data copy. Run the notebook to regenerate the complete comparison.

### MLflow Tracking

Each model evaluation starts an MLflow run named after the model. The notebook logs:

- Model name and parameters
- Train and test R²
- MSE and RMSE
- Cross-validation mean and standard deviation
- Overfitting gap
- The fitted scikit-learn model

The notebook currently uses MLflow's default tracking configuration. For a persistent tracking server, set an explicit tracking URI before running the evaluation cells.

## Deep Learning Workflow

The deep-learning experiment uses PyTorch for tabular regression.

### Preprocessing

The DL pipeline prepares the data as follows:

1. Load the OpenML Ames Housing data.
2. Separate numerical and categorical features.
3. Impute numerical missing values with the median.
4. Impute categorical missing values with the most frequent category.
5. One-hot encode categorical features.
6. Standardize numerical features.
7. Apply `log1p` to the target price so the network learns a less skewed target distribution.
8. Create PyTorch datasets and `DataLoader` objects with a batch size of 64.

The encoded feature matrix is numeric before it is converted into PyTorch tensors. This is necessary because PyTorch layers cannot consume raw string or pandas categorical values.

### Neural Network

The network is a multilayer perceptron for tabular data:

```text
Input features
	-> Linear(input_dim, 128)
	-> ReLU
	-> Linear(128, 64)
	-> ReLU
	-> Linear(64, 32)
	-> ReLU
	-> Linear(32, 1)
```

Training uses:

- AdamW optimization
- Mean squared error loss
- Mini-batches of 64 observations
- Gradient clipping
- Learning-rate reduction when validation loss stops improving
- Early stopping
- CPU execution in the recorded notebook run

The recorded training run stopped early at approximately epoch 109. The notebook includes a training-versus-validation loss plot so that convergence and overfitting can be inspected visually.

### DL Evaluation Status

The neural-network training workflow is implemented, but the final notebook cells should be used to calculate and report test-set MAE, MSE, RMSE, and R² after training. A complete comparison should evaluate the model on the original price scale by applying the inverse of the target transformation:

```python
predicted_price = np.expm1(predicted_log_price)
```

This makes the neural-network metrics directly interpretable in price units.

## Visualizations

The notebooks include visual analysis such as:

- Feature-correlation heatmaps
- Sale-price distributions
- Actual-versus-predicted price plots
- Random Forest feature-importance charts
- Training and validation loss curves for the neural network

These plots help identify skewed targets, influential predictors, prediction quality, and possible overfitting.

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install \
	pandas numpy matplotlib seaborn scikit-learn \
	xgboost lightgbm catboost mlflow \
	torch torchvision torchaudio jupyter
```

The exact PyTorch installation may differ depending on whether CPU or CUDA support is required. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for a GPU-specific command.

## Running the Project

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Then open the relevant notebook:

1. Open `House_prices_ML/app.ipynb` for the classical ML experiment.
2. Open `House_prices_DL/app.ipynb` for the PyTorch experiment.
3. Run the cells from top to bottom.
4. Review the printed metrics and plots.

For the nested ML notebook, the expected working location is:

```text
House_prices_ML/House Prices model/code/
```

This matters because the notebook uses a relative path to `../data/AmesHousing.csv`.

## Reproducibility

For repeatable experiments:

- Run cells in order from a fresh kernel.
- Keep `random_state=42` for the ML train/test split.
- Record Python and package versions.
- Use the same dataset copy when comparing models.
- Keep preprocessing inside the training pipeline to avoid data leakage.
- Evaluate transformed-target models after converting predictions back to the original price scale.

The repository does not currently include a locked `requirements.txt`, `environment.yml`, or saved model artifacts. Adding one of these would make the experiments easier to reproduce on another machine.

## Current Limitations and Improvements

### Machine Learning

- The main ML experiment uses only eight numerical features, even though the dataset contains many more useful variables.
- Categorical features are not included in the classical feature matrix.
- Some preprocessing is written manually rather than using a single `ColumnTransformer` and `Pipeline`.
- Hyperparameter search is imported or explored but is not consistently used for every model.
- The MLflow tracking URI and experiment name are not explicitly configured.

### Deep Learning

- The DL experiment has a stronger categorical preprocessing strategy, but its final test metrics should be reported explicitly.
- The neural network is a baseline MLP; more extensive tuning could improve performance.
- A controlled comparison should use the same train/test split and the same source dataset as the classical ML models.
- The project would benefit from saving the preprocessing objects and model weights for later inference.

### Recommended Next Steps

1. Build one shared preprocessing pipeline for both ML and DL experiments.
2. Use all meaningful features, including categorical variables.
3. Add a fixed validation protocol and a final untouched test set.
4. Complete the DL test evaluation with MAE, RMSE, and R² on the original price scale.
5. Add a `requirements.txt` or `environment.yml` file.
6. Save the best model and preprocessing pipeline for inference.
7. Add prediction examples for new house records.

## License and Data Attribution

This repository is an educational project. The Ames Housing data is provided through the linked Kaggle/OpenML sources and remains subject to their respective terms of use.
