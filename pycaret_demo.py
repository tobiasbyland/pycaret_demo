

# -------------------------------------------------------
# built with:
# python version 3.10
# req.txt from pycaret's github
# pycaret 3.1.0
# -------------------------------------------------------


# ====================================
# Auto-ML for a classification problem
# ====================================

from pycaret.datasets import get_data

# sort by rand before setup?

# ---------
# parameters
# ---------

p_target = 'CLASS'
p_name_pipeline = 'finalmodel'

# get_data('index')
data_mod = get_data('poker')[1:90000]
data_new = get_data('poker')[90001:100000]


# -------------------------------------------------------------------------------------
# Auto ML
# -------------------------------------------------------------------------------------

from pycaret.classification import load_experiment
from pycaret.classification import *

# Setup up modelling process
# --------------------------

# Possible configurations: https://pycaret.gitbook.io/docs/get-started/preprocessing/data-preparation
setup(data_mod
      , target = p_target
      , fold_strategy = 'stratifiedkfold' # Choice of cross-validation strategy
      , fold = 10 # The number of folds to be used in cross-validation.
      , fix_imbalance = False # When set to True, the training dataset is resampled using the algorithm defined in fix_imbalance_method
      , remove_outliers = True # When set to True, outliers from the training data are removed using an Isolation Forest
      , normalize = True # When set to True, the feature space is transformed using the method defined under the normalized_method parameter
      , transformation = True # When set to True, a power transformer is applied to make the data more normal / Gaussian-like.
      , polynomial_features = True # When set to True, new features are created based on all polynomial combinations that exist within the numeric features in a dataset to the degree defined in the polynomial_degree parameter
      , group_features = None # When a dataset contains features that have related characteristics, the group_features param can be used for statistical feature extraction
      , bin_numeric_features = None # When a list of numeric features is passed they are transformed into categorical features using K-Mean
      , feature_selection = True # When set to True, a subset of features is selected based on a feature importance score determined by feature_selection_estimator
      , remove_multicollinearity = True # When set to True, features with the inter-correlations higher than the defined threshold are removed
      , low_variance_threshold = 0 # Remove features with a training-set variance lower than the provided threshold. If 0, keep all features with non-zero variance, i.e. remove the features that have the same value in all samples
      , log_experiment = True #  Setting to True will use MLFlow
      , experiment_name = 'pycaret demo - classif'
      )
 
# Find the best model
# -------------------

best = compare_models()

# mlflow ui # <-- run from the command line

# describe best model on test data
# --------------------------------

plot_model(best, plot = 'confusion_matrix'
           #, plot_kwargs = {'percent' : True}
           )
plot_model(best, plot = 'auc')
plot_model(best, plot = 'feature')
plot_model(best, plot = 'lift')

# perform hyperparameter tuning
# -----------------------------

# The tuning grid for hyperparameters is already defined by PyCaret for all the models in the library. 
# However, if you wish you can define your own search space
# https://pycaret.gitbook.io/docs/get-started/functions/optimize#tune_model
tuned = tune_model(best
                   , choose_better = True # A tuned model does not always deliver better results.  If this is set to True, an improved model is always returned
                   , n_iter = 10 # number of iterations, which eventually depends on how much time and resources you have available
                   , optimize = 'Accuracy'
                   ) # check further settings
# pull()


# bagging/boosting
# ----------------
# todo: https://michael-fuchs-python.netlify.app/2022/01/01/automl-using-pycaret-classification/#tune-the-model


# finalize the best model - train the best model on the entire dataset, including the test set 
# ----------------------------------------------------------------------------------------------

final_model = finalize_model(tuned)

dashboard(final_model)

# save model to disk
# ------------------

save_model(final_model, p_name_pipeline)


# -------------------------------------------------------------------------------------
# predict on new data
# -------------------------------------------------------------------------------------

# load pipeline
data_new.drop(p_target, axis = 1, inplace = True)
data_new_pred = predict_model(final_model, raw_score = True, data = data_new)

import plotly.express as px
fig = px.histogram(data_new_pred, x="prediction_score_1")
fig.show()


load_model(p_name_pipeline)
load_experiment(p_name_pipeline, data = data_new)
# https://pycaret.gitbook.io/docs/get-started/functions/deploy#predict_model
