import pandas as pd
import seaborn as sns
from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle


---------------------------------------------------------------------------------------------------------------------

dataset_url= 'https://www.kaggle.com/siddharthss/crop-recommendation-dataset'
od.download('https://www.kaggle.com/siddharthss/crop-recommendation-dataset')
