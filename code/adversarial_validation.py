import pandas as pd
import numpy as np

from copy import deepcopy

from typing import Dict, Callable, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

import shap

from tqdm.notebook import tqdm as tqdm

from xgboost import XGBClassifier

import pdb


class AdversarialValidation:

    """ """

    def __init__(
        self,
        features: list = [],
        base_estimator: Callable = None,
        time_column: str = None,
        train_end_date: str = None,
        is_train_column: str = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Creates an instance of the class AdversarialValidation. This will recieve either a dataframe with a time column and a training end date to make the splits
        or a frame with a column that indicates wheter a samples comes from training or testing, this in case we already have the splits.

        Once received, It will create a ramdom split and will train a binary classfier to separate both samples.

        Args:
            frame (DataFrame): The dataframe that contains the features and time column to create the adversarial model's
            dataset.
            base_estimator (Callable): A Scikit-learn classfier with a predict_proba method.
            time_column (str): The column that indicates the date associated to a sample.
            train_end_date (str): The date that will be used to split the data if there is a date column.
            is_train_column (str): The column that indicates whether a sample comes from training (0) or testing (1) in case no time column is provided.
            test_size (float): The size of the test set for the random split.
            random_state (int): Seed used for the data split.

        Returns:
            tuple: DataFrames X and y for training and testing
        """

        self.features = features
        self.base_estimator = base_estimator
        self.time_column = time_column
        self.train_end_date = train_end_date
        self.is_train_column = is_train_column
        self.test_size = test_size
        self.random_state = random_state

    def _make_split(self, frame: pd.DataFrame, feature_names: list = None) -> Tuple:
        """
        Creates a target column based upon training end date. It labels a sample as 1 if its date column is after
        the training period and 0 otherwise.

        Args:
            frame (DataFrame): The dataframe that contains the features and time column to create the adversarial model's
            dataset.

        Returns:
            tuple: DataFrames X and y for training and testing
        """

        if not feature_names:
            feature_names = self.features[:]

        if self.time_column:
            frame.loc[:, "target"] = np.where(
                frame[self.time_column] < self.train_end_date, 0, 1
            )

            X_train, X_test, y_train, y_test = train_test_split(
                frame[feature_names],
                frame["target"],
                test_size=self.test_size,
                random_state=self.random_state,
            )
        else:
            frame.rename(columns={self.is_train_column: "target"}, inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(
                frame[feature_names],
                frame["target"],
                test_size=self.test_size,
                random_state=self.random_state,
            )
        # pdb.set_trace()

        self.train_class_balance_ = np.mean(y_train)
        self.test_class_balance_ = np.mean(y_test)

        assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

        return X_train, X_test, y_train, y_test

    def _estimate_performance(self, y_true, y_score):
        """


        Returns the ROC AUC of the model on the test set.

        Args:
            y_true (pd.Series): Series of shape (n_samples,) containing the true labels.
            y_score (pd.Series): Series of shape (n_samples,) containing the predicted probabilities.
        Returns:
            roc_auc_score (float): value containing the area under the roc curve.

        """

        return roc_auc_score(y_true=y_true, y_score=y_score)

    def _construct_estimator(self, params: dict = {"random_state": 42}):
        """ """

        base_estimator = XGBClassifier(**params)

        return base_estimator

    def _estimate_shap_values(self, estimator: Callable = None, X: pd.DataFrame = None):

        """ """

        explainer = shap.TreeExplainer(model=estimator)
        shap_values = explainer.shap_values(X)

        return shap_values

    def fit(
        self,
        frame: pd.DataFrame = pd.DataFrame([]),
        estimator_params: dict = {"random_state": 42, "enable_categorical": True},
    ):
        """
        Trains a binary classifier to separate training from testing samples.

        Args:
            frame (DataFrame): The dataframe that contains the features and time column to create the adversarial model's
            dataset.
            estimator_params (Dictionary): A dictionary containing the hyperparameters to be passed to the based estimator.
        Returns:
            : None


        """

        X_train, X_test, y_train, y_test = self._make_split(
            frame, feature_names=self.features
        )

        if self.base_estimator is None:
            self.base_estimator = self._construct_estimator(
                params=estimator_params
            ).fit(X_train, y_train)

        predictions = self.predict_proba(X_test)[:, 1]

        self.performance_ = self._estimate_performance(y_test, predictions)

        self.shap_values = self._estimate_shap_values(self.base_estimator, X_test)

        print(f"init_performance_:{self.performance_}")

        return self

    def predict(self, frame, threshold: float = 0.5):

        """
        Makes inference on the passed dataframe to predict whether a sample comes from train or test set.
        Args:
            frame (DataFrame): The dataframe that contains the features  to predict if the samples come from
            train or test sample
            threshold (float): Value to assign a sample to the positive class.

        Returns:
            np.array: A (n_samples,) array containing the predicted value according to the default threshold (0.5).

        """

        predictions = self.predict_proba(frame[self.features])[:, 1]
        predictions = np.where(predictions > threshold, 1, 0)

        return predictions

    def predict_proba(self, frame):

        """
        Makes inference on the passed dataframe to predict whether a sample comes from train or test set.
        Args:
            frame (DataFrame): The dataframe that contains the features  to predict if the samples come from
            train or test sample

        Returns:
            np.array: A (n_samples, 2) array containing the predicted probability.
        """

        predictions = self.base_estimator.predict_proba(frame[self.features])

        return predictions

    def plot_shap_values(self, plot_type: str = "dot", max_display: int = 25):

        """
        Creates a plot of the SHAP values for the model

        Args:
            plot_type (string): A string containing the type of plot to be created, it may be either dot or bar kind.

        Returns:
            plt.figure: A Matplotlib figure containing the SHAP plot.


        """

        shap.summary_plot(
            self.shap_values,
            feature_names=self.features,
            plot_type=plot_type,
            max_display=max_display,
        )

    def recursive_feature_elimination(
        self,
        frame: pd.DataFrame,
        n_features_remove: int = None,
        threshold_remove_until: float = 0.5,
    ):

        """

        Iteratively removes features from the original dataset and refits the adversarial model to measure the model's
        performance degradation after removal.

        Starts from the most the the least important feature according to the SHAP values.

        Args:
            frame (DataFrame): The dataframe that contains the features  to predict if the samples come from
            train or test sample.
            n_features_remove: (int): Number of features to be removed after stop.
            threshold_remove_until (float): Percentage of features to be removed.


        Returns:
            DataFrame: A DataFrame of shape (n_features_removed, 2) containing the mmodel performance after every features' removal.

        """

        ldf = list()
        tmp_performance: float = 1.0

        all_features_shap = pd.DataFrame(
            {"average_shap_value": np.abs(self.shap_values).mean(axis=0)},
            index=self.features,
        ).sort_values(by="average_shap_value", ascending=False)
        all_features_adversarial_performance = self.performance_
        most_important_feature: str = all_features_shap.idxmax().values[0]
        print(f"most_important_feature: {most_important_feature}")
        # pdb.set_trace()

        ldf.append(("all_features", all_features_adversarial_performance))

        tmp_features = self.features[:]

        if n_features_remove is not None:
            for iter in tqdm(range(n_features_remove)):
                tmp_frame = deepcopy(frame)
                tmp_features.remove(most_important_feature)
                X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = self._make_split(
                    tmp_frame, feature_names=tmp_features
                )
                tmp_estimator = self._construct_estimator(params={"random_state": 42})
                tmp_estimator = tmp_estimator.fit(X_train_tmp, y_train_tmp)
                predictions_tmp = tmp_estimator.predict_proba(X_test_tmp)[:, 1]
                tmp_shap_values = self._estimate_shap_values(tmp_estimator, X_test_tmp)

                tmp_features_shap = pd.DataFrame(
                    {"average_shap_value": np.abs(tmp_shap_values).mean(axis=0)},
                    index=tmp_features,
                ).sort_values(by="average_shap_value", ascending=False)
                most_important_feature: str = tmp_features_shap.idxmax().values[0]
                print(f"most_important_feature: {most_important_feature}")
                tmp_performance = self._estimate_performance(
                    y_test_tmp, predictions_tmp
                )
                print(f"tmp_performance:{tmp_performance}")

                ldf.append((most_important_feature, tmp_performance))
        elif threshold_remove_until is not None:
            while (tmp_performance > threshold_remove_until) and len(tmp_features) > 0:
                tmp_frame = deepcopy(frame)
                tmp_features.remove(most_important_feature)
                X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = self._make_split(
                    tmp_frame, feature_names=tmp_features
                )
                tmp_estimator = self._construct_estimator(params={"random_state": 42})
                tmp_estimator = tmp_estimator.fit(X_train_tmp, y_train_tmp)
                predictions_tmp = tmp_estimator.predict_proba(X_test_tmp)[:, 1]
                tmp_shap_values = self._estimate_shap_values(tmp_estimator, X_test_tmp)

                tmp_features_shap = pd.DataFrame(
                    {"average_shap_value": np.abs(tmp_shap_values).mean(axis=0)},
                    index=tmp_features,
                ).sort_values(by="average_shap_value", ascending=False)
                most_important_feature: str = tmp_features_shap.idxmax().values[0]
                print(f"most_important_feature: {most_important_feature}")
                tmp_performance = self._estimate_performance(
                    y_test_tmp, predictions_tmp
                )
                print(f"tmp_performance:{tmp_performance}")

                ldf.append((most_important_feature, tmp_performance))

        return pd.DataFrame(ldf, columns=["columns", "adversarial_model_performance"])

    def save_model(
        self,
        model_path: str = "../model-files/",
        model_name: str = "adversarial_classfier",
    ):
        """
        Saves the trained adversarial model to the specified path as a pickle file.

        Args:
            model_path (str): String with the path to save the model to.
            model_name: (str): String with the name to save the model.

        Returns:
            None


        """

        pickle.dump(self.base_estimator, open(f"{model_path}{model_name}.pickle", "wb"))
