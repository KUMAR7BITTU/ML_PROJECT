from collections import namedtuple
from datetime import datetime
import uuid
from housing.config.configuration import Configuartion
from housing.logger import logging, get_log_file_name
from housing.exception import HousingException
from threading import Thread
from typing import List

from multiprocessing import Process
from housing.entity.artifact_entity import ModelPusherArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from housing.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from housing.entity.config_entity import DataIngestionConfig, ModelEvaluationConfig
from housing.component.data_ingestion import DataIngestion
from housing.component.data_validation import DataValidation
from housing.component.data_transformation import DataTransformation
from housing.component.model_trainer import ModelTrainer
from housing.component.model_evaluation import ModelEvaluation
from housing.component.model_pusher import ModelPusher
import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd
from housing.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp","running_status", "start_time", "stop_time", "execution_time", "message","experiment_file_path", "accuracy", "is_model_accepted"])
# Whenever we will create object of pipeline then that information we will store in initialization_timestamp .




class Pipeline(Thread):
    # We have converted this pipeline into threads. So, i have pipeline then i made this thread so, i can start my training using thread . We just want to do our training seperately . When you run your application then main Thread will be responsible for to execute your entire code . We are forking one more Thread from the main . Thatmeans we are creating one more branch from the thread which will takecare of execution of training .It is independent to the main Thread . So for example intially two developers were working on single project but only one person was writing code . Now we have creating such a system where both of the person can write code .
    experiment: Experiment = Experiment(*([None] * 11))
    # The above line of code tell us that in Experiment namedtuple we have all different-different attribute so i am trying to initialize all of them as None .
    # None and 0 are two different things . 0 is a value and None is a datatype .
    
    # The idea behind above code is if my training pipeline is already being run then i will not start new training pipeline . If running_status will be true , we will say no to new request of running pipeline .
    
    #experiment_file_path = os.path.join(config.training_pipeline_config.artifact_fir,EXPERIMENT_DIR_NAME,EXPERIMENT_FILE_NAME)
    # experiment_file_path will show us record of our experiments (pipeline experiment).
    
    # Use of class level attribute is that it is same for every object . 
    
    experiment_file_path = None

    def __init__(self, config: Configuartion ) -> None:
        try:
            # Whenever i will create object of pipeline then it will automatically create artifact folder .
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) \
            -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise HousingException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         data_transformation_artifact=data_transformation_artifact
                                         )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise HousingException(e, sys) from e
    
    # Here we are just allowing pipeline to run seperately . we are not allowing training to run parallely .
    def run_pipeline(self):
        try:
            # If running_status is true means the pipeline is already running . we will not again start the pipeline rather we will return same the same pipeline  which is running . By writing this line code we are trying not to run another pipeline .
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            # data ingestion
            logging.info("Pipeline starting.")

            experiment_id = str(uuid.uuid4())
            # It will generate the global unique id and it will be generated randomly .
            
            # If no pipeline is being running previously then only we will come to this line of code .  we will provide the experiment details in this Experiment named tuple .
            Pipeline.experiment = Experiment(experiment_id=experiment_id,initialization_timestamp=self.config.time_stamp,artifact_time_stamp=self.config.time_stamp,running_status=True,start_time=datetime.now(),stop_time=None,execution_time=None,experiment_file_path=Pipeline.experiment_file_path,is_model_accepted=None,message="Pipeline has been started.",accuracy=None,)
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            # Bydefault i am going to start my pipeline so i will make running_status=True .

            self.save_experiment()
            # This save_experiment() function will write all the datas that we have initialized above in experiment.csv file .

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact,model_trainer_artifact=model_trainer_artifact)

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")

            stop_time = datetime.now()
            # When the above line will be executed then i can say my pipeline has been completed . I can take the stop_time .
            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,initialization_timestamp=self.config.time_stamp,artifact_time_stamp=self.config.time_stamp,running_status=False,start_time=Pipeline.experiment.start_time,stop_time=stop_time,execution_time=stop_time - Pipeline.experiment.start_time,message="Pipeline has been completed.",experiment_file_path=Pipeline.experiment_file_path,is_model_accepted=model_evaluation_artifact.is_model_accepted,accuracy=model_trainer_artifact.model_accuracy)
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
            # Then we will again save the record . So, we will save every record two times when the pipeline has been started and when the pipeline has been stoped .
        except Exception as e:
            raise HousingException(e, sys) from e

    def run(self):
        # When we start Thread then we this run() function is going to be called . And within run() function we are calling self.run_pipeline() . So, whenever i will call start function for any object of this pipeline class then it will call this run() function .
        try:
            self.run_pipeline()
        except Exception as e:
            raise e

    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]})

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path, mode="w", index=False, header=True)
            else:
                print("First start experiment")
        except Exception as e:
            raise HousingException(e, sys) from e

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1 * int(limit)
                return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise HousingException(e, sys) from e
