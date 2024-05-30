import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Import libs to solve classififcation task
import lightgbm as lgb

# Define optimal threshold
MODEL_TH = 0.334

# Visualization of predictions
def make_visual(pred):
    bins_range = np.linspace(0, 1, 16)
    plt.figure(figsize=(10, 6))
    plt.vlines(MODEL_TH, ymax=300_000, ymin=0, label="Threshold of our model", colors='r', linestyles='dashed')
    plt.hist(pred, bins=bins_range, edgecolor='black')
    plt.title("Model predictions based on test data")
    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.legend()

    return plt

# Extraction function for top 5 features
def extract_feature(model):
    feature_importances = {int(importance) for importance in model.feature_importance()}
    feature_names = model.feature_name()
    features = list(zip(feature_names, feature_importances))
    sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

    # Select to5 feature
    top5_features = sorted_features[:5]

    top5_features_dict = {feature: importance for feature, importance in top5_features}
    top5_features_json = json.dumps(top5_features_dict, indent=4)

    return top5_features_json

# Make prediction
def make_pred(dt, path_to_file):
    
    print('Importing pretrained model...')
    model = lgb.Booster(model_file="./models/my_lightgbm_model.txt")

    pred = model.predict(dt, raw_score=False)

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (pred > MODEL_TH) * 1 
        })
    print('Prediction complete!')

    # Make histogram predict in float form digit
    hist_visual = make_visual(pred) 

    # Extract top5 feature importance
    feature_importance = extract_feature(model)

    # Return proba for positive class
    return submission, hist_visual, feature_importance

