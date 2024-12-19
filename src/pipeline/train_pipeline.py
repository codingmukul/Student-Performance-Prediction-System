import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training(self):
        """
        Execute the complete training pipeline
        """
        try:
            logging.info(">>>>>>> Starting Training Pipeline <<<<<<<<")

            # Step 1: Data Ingestion
            logging.info(">>>>>>> Starting Data Ingestion <<<<<<<<")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")

            # Step 2: Data Transformation
            logging.info(">>>>>>> Starting Data Transformation <<<<<<<<")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
            logging.info("Data transformation completed")
            logging.info(f"Preprocessor object saved at: {preprocessor_path}")

            # Step 3: Model Training and Evaluation
            logging.info(">>>>>>> Starting Model Training <<<<<<<<")
            model_score = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )
            logging.info(f"Model training completed. R2 Score: {model_score}")

            logging.info(">>>>>>> Training Pipeline Completed <<<<<<<<")
            return model_score

        except Exception as e:
            logging.error("Error occurred in training pipeline")
            raise CustomException(e, sys)

def main():
    try:
        pipeline = TrainingPipeline()
        score = pipeline.start_training()
        
        logging.info("=" * 50)
        logging.info(f"Final Model Performance (R2 Score): {score}")
        logging.info("=" * 50)
        
        # Print results to console as well
        print("\n" + "="*50)
        print("Training Pipeline Completed Successfully!")
        print(f"Model Performance (R2 Score): {score}")
        print("="*50 + "\n")
        
        return score

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()