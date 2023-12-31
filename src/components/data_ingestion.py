import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

## Initialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

## Create data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingetion_config=DataIngestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Starts')
        
        try:
            df=pd.read_csv(os.path.join('notebooks/data','card_data.csv'))
            logging.info('Read Dataset as Pandas DataFrame')
            
            # create artifacts directory is not already exists
            
            os.makedirs(os.path.dirname(self.ingetion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingetion_config.raw_data_path,index=False)
            
            # Now slipt the data
            logging.info('Train Test Split')
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            
            train_set.to_csv(self.ingetion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingetion_config.test_data_path,index=False,header=True)
            
            logging.info('Data Ingestion is completed')
            
            return(
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path
            )
        
        except Exception as e:
            logging.info('Error occured in Data Ingestion config')
        