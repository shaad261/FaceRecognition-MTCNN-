from src.logger import logging
from src.exception import CustomException
from src.predictor.facePredictor import FacePredictor 

import sys
if __name__=="__main__":
    logging.info("The execution has started")

    try:
        
        faceprediction=FacePredictor()
        faceprediction.detectFace()
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

