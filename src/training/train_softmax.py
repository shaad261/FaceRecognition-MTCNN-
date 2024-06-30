from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import argparse
import pickle
import os
from keras.utils import to_categorical

# Construct the argumet parser and parse the argument
from src.detectfaces_mtcnn.Configurations import get_logger
from src.training.softmax import SoftMax


class TrainFaceRecogModel:

    def __init__(self, args):

        self.args = args
        self.logger = get_logger()
        # Load the face embeddings
        self.data = pickle.loads(open("src/faceEmbeddingModels/embeddings.pickle", "rb").read())

    def trainKerasModelForFaceRecognition(self):
        if len(self.data["names"]) == 0 or len(np.unique(self.data["names"])) == 0:
            print("Error: No valid labels found in self.data['names']. Cannot proceed with training.")
            return  # Exit the function or handle the error condition

        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)

        if num_classes == 0:
            print("Error: No unique classes found in labels. Cannot proceed with training.")
            return  # Exit the function or handle the error condition

        one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
        labels = one_hot_encoder.fit_transform(labels)

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        # Build softmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # Create KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=True)

        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
        
        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]

            his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            verbose=1, validation_data=(X_val, y_val))

            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

            self.logger.info(his.history['acc'])
        # Ensure the directory exists before saving the model
        model_dir = os.path.dirname(self.args['model'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the face recognition model
        model.save(self.args['model'])

        # Save the LabelEncoder
        with open(self.args["le"], "wb") as f:
            pickle.dump(le, f)
