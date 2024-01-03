from datetime import datetime
import os

MONGO_DATABASE_NAME = "Credit-Card"
MONGO_COLLECTION_NAME = "Card"
MONGO_DB_URL =  "mongodb+srv://ankitbodar001:p317IcjArXGPWNKj@cluster0.1vyd668.mongodb.net/?retryWrites=true&w=majority"

TARGET_COLUMN = "default payment next month"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"