from metaflow import (
    FlowSpec,
    card,
    step,
    project,
    pypi_base,)
from metaflow.exception import MetaflowException
import logging
from data_utility import configure_logging

configure_logging()

@project(name="deploy")
@pypi_base(
        python="3.10.9",
        packages={'numpy': '1.26.4',
                  'seaborn': '0.12.2',
                'mlflow': '3.1.4',
                'pandas': '2.3.1',
                'pyarrow': '14.0.2',
                'scikit-learn': '1.7.1'})
class Train(FlowSpec):
    """
    This pipeline trains, logs and deploys a random forrest classifier on the
    penguins dataset.
    """
    @step
    def start(self):
        """
        Download the dataset if it does not already exist.
        Configure MLflow and set the run ID as an environment variable.
        """
        import os
        import mlflow
        from datetime import datetime

        # Set  MLflow tracking URI and experiment
        self.parent_run_name = \
            f"penguins-{datetime.now().strftime('%d/%m/%Y_%H:%M:%S')}"
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5001")
            mlflow.set_experiment("penguins-classification")
            with mlflow.start_run(run_name=self.parent_run_name) as run:
                self.parent_run_id = run.info.run_id
        except Exception as e:
            message = (
                f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            )
            raise RuntimeError(message) from e

        logging.info(
            "MLFLOW_RUN_ID environment variable set as: %s",
            self.parent_run_id,
        )

        self.next(self.download_dataset)

    @step
    def download_dataset(self):
        import mlflow
        import os
        from data_utility import download_penguins
        import pandas as pd

        # Download and persist the penguins dataset
        self.data_directory = "data"
        self.data_filename = "penguins.csv"
        self.path = f"{self.data_directory}/{self.data_filename}"
        if not os.path.isdir(self.data_directory):
            download_penguins()

        # Convert the dataset into a dataframe
        self.data = pd.read_csv(self.path)
        self.mlflow_dataset = mlflow.data.from_pandas(
            self.data, name="seaborn-penguins", targets="species"
        )

        # log the dataset used in the run
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("penguins-classification")
        with mlflow.start_run(run_id=self.parent_run_id):
            mlflow.log_input(self.mlflow_dataset, context="training")
            mlflow.log_param("dataset_shape_rows", self.data.shape[0])
            mlflow.log_param("dataset_shape_cols", self.data.shape[1])
            mlflow.log_param("n_species", self.data['species'].nunique())

        self.next(self.validate)

    @step
    def validate(self):
        """
        Validate the `self.data` DataFrame before transforming it.

        Checks performed:
        1. `self.data` is a pandas DataFrame.
        2. `self.data` contains all required columns.
        """

        import pandas as pd

        required_cols = {
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        }

        # Validate DataFrame type
        logging.info("Validating: checking if self.data is a pandas DataFrame...")
        if not isinstance(self.data, pd.DataFrame):
            raise MetaflowException(
                "Validation failed: self.data is not a pandas DataFrame."
            )
        logging.info("✅ self.data is a valid pandas DataFrame.")

        # Validate required columns
        logging.info("Validating: checking required columns in DataFrame...")
        self.missing_columns = required_cols - set(self.data.columns)
        if self.missing_columns:
            raise MetaflowException(
                f"Validation failed: Missing required columns: "
                f"{', '.join(sorted(self.missing_columns))}."
            )
        logging.info("✅ DataFrame contains all required columns.")

        self.next(self.transform_data)

    @step
    def transform_data(self):
        """
        Split data into train and test datasets
        """
        from data_utility import split_data

        # Create test and train features and labels from the dataset
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test,
         ) = split_data(self.data)

        self.next(self.build_pipeline)

    @step
    def build_pipeline(self):
        """
        Build a scikit-learn pre-processing pipeline with Random Forest
        base estimator.
        """
        from training_utility import build_sklearn_pipeline

        self.base_pipeline = build_sklearn_pipeline(self.X_train)

        self.next(self.generate_param_combinations)

    @step
    def generate_param_combinations(self):
        """
        Generate random parameter combinations for hyperparameter search.
        """
        import random

        logging.info(f"Generating parameter combinations...")

        # Define parameter distributions
        param_distributions = {
            'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
            'classifier__max_depth': [None, 5, 10, 15, 20, 25, 30],
            'classifier__min_samples_split': [2, 3, 4, 5, 8, 10],
            'classifier__min_samples_leaf': [1, 2, 3, 4, 5],
            'classifier__max_features': ['sqrt', 'log2', None, 0.5, 0.7],
            'classifier__bootstrap': [True, False]
        }

        # Generate random parameter combinations
        random.seed(42)
        self.param_combinations = [
            {**{param_name: random.choice(param_values)
                for param_name, param_values in param_distributions.items()},
             "trial_id": i}
            for i in range(6)
        ]
        logging.info("Generated parameter %s combinations",
                     len(self.param_combinations))

        self.next(self.cross_validation, foreach='param_combinations')


    @step
    def cross_validation(self):
        """
        Evaluate a single parameter combination using cross-validation.
        This step runs in parallel for each parameter combination.
        """
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score
        import numpy as np
        import mlflow

        # Get current parameter combination
        params = self.input
        trial_id = params.pop('trial_id')
        run_name = f"Trail {trial_id}"

        logging.info("Evaluating trial %s with parameters: %s",
                     trial_id, params )

        # Clone the base pipeline and set parameters
        pipeline = clone(self.base_pipeline)
        pipeline.set_params(**params)

        # Perform cross-validation
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("penguins-classification")
        with mlflow.start_run(run_id=self.parent_run_id):
            with mlflow.start_run(nested=True, run_name=run_name):

                # enable autologging
                mlflow.sklearn.autolog(log_models=False)

                cv_scores = cross_val_score(
                    pipeline, self.X_train, self.y_train,
                    cv=3, scoring='accuracy', n_jobs=1
                )

                # Fit the model and evaluate on test set
                pipeline.fit(self.X_train, self.y_train)
                test_predictions = pipeline.predict(self.X_test)
                test_accuracy = accuracy_score(self.y_test, test_predictions)

                # Store results for future steps
                self.trial_results = {
                    'trial_id': trial_id,
                    'parameters': params,
                    'cv_scores': cv_scores,
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores),
                    'test_accuracy': test_accuracy,
                    'fitted_model': pipeline,
                }


        logging.info(
            "Trial %s: CV Score = %.4f ± %.4f, Test Accuracy = %.4f",
            trial_id,
            self.trial_results['mean_cv_score'],
            self.trial_results['std_cv_score'],
            test_accuracy
        )
        self.next(self.collect_results)

    @step
    def collect_results(self, inputs):
        """
        Collect results from all parallel evaluations, find the best model
        and log it to the parent run
        """
        import mlflow
        import numpy as np
        from mlflow.models import infer_signature

        logging.info("Collecting results from all trials...")

        # Collect all trial results
        self.all_results = []
        for input_data in inputs:
            self.all_results.append(input_data.trial_results)

        # Sort by mean CV score (descending)
        self.all_results.sort(key=lambda x: x['mean_cv_score'], reverse=True)

        # Get best model
        self.best_result = self.all_results[0]

        logging.info(f"Best parameters %s:",
                     self.best_result['parameters'])
        logging.info(
            "Best CV score: %.4f ± %.4f",
            self.best_result['mean_cv_score'],
            self.best_result['std_cv_score']
        )
        logging.info(
            "Best test accuracy: %.4f",
            self.best_result['test_accuracy']
        )

        # Log summary results to MLflow parent run
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("penguins-classification")
        with mlflow.start_run(run_id=inputs[0].parent_run_id):
            # Log best model results
            mlflow.log_params(self.best_result['parameters'])
            mlflow.log_metric("best_cv_score",
                              self.best_result['mean_cv_score'])
            mlflow.log_metric("best_cv_std",
                              self.best_result['std_cv_score'])
            mlflow.log_metric("best_test_accuracy",
                              self.best_result['test_accuracy'])

            # Log summary statistics
            cv_scores = [result['mean_cv_score'] for result in self.all_results]
            test_scores = [result['test_accuracy'] for result in self.all_results]
            mlflow.log_metric("avg_cv_score", np.mean(cv_scores))
            mlflow.log_metric("avg_test_accuracy", np.mean(test_scores))
            mlflow.log_metric("trials_completed", len(self.all_results))

            # Log the best model
            logging.info("Logging best model from trail: %s",
                         self.best_result["trial_id"])
            self.X_train = inputs[0].X_train
            signature = infer_signature(
                self.X_train,
                self.best_result['fitted_model'].predict(self.X_train)
            )
            mlflow.sklearn.log_model(
                sk_model=self.best_result['fitted_model'],
                artifact_path="best_model",
                signature=signature
            )

        self.next(self.end)

    @step
    def end(self):
        """End training pipeline"""
        logging.info(msg="Pipeline completed successfully!!!")

if __name__ == '__main__':
    Train()





























