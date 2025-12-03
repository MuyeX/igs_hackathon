import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.statespace.varmax import VARMAX
from darts.models import BlockRNNModel, TransformerModel, RandomForest, VARIMA, KalmanFilter, MovingAverageFilter
from darts.ad import DifferenceScorer, OrAggregator, ThresholdDetector, NormScorer
from darts.ad.detectors.iqr_detector import IQRDetector
from darts.ad import ForecastingAnomalyModel
import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

random.seed(42)


class ForecastingAnomaly():
    def __init__(self, model='RNN', scorer='difference', input_chunk_length=6, training_length=7, epochs=20):
        self.model_name = model

        if model == 'RNN' or model == 'LSTM' or model == 'GRU':
            self.model = BlockRNNModel(
                    model=model,
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=(training_length-input_chunk_length),
                    n_epochs=epochs,
                    model_name=model
                )

        self.scorer = NormScorer(component_wise=True)
        self.anomaly_model = ForecastingAnomalyModel(model=self.model, scorer=self.scorer)
        self.detector = ThresholdDetector(high_threshold=1.2, low_threshold=0)
        self.aggregator = OrAggregator()

        self.stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.05,
            mode='min'
        )

    def update_model(self, model):
        self.anomaly_model = ForecastingAnomalyModel(model=model, scorer=self.scorer)

    def update_detector(self, ts):
        self.detector.fit(ts)

    def anomaly_detection(self, series_list, covar_list=None):
        predictions = self.anomaly_model.predict_series(series=series_list, past_covariates=covar_list)
        anomaly_score = self.scorer.score_from_prediction(series=series_list, pred_series=predictions)
        binary_predictions = self.detector.detect(anomaly_score)
        if len(binary_predictions[0].components) == 1:
            abnormal = binary_predictions
        else:
            abnormal = self.aggregator.predict(binary_predictions)

        return abnormal
