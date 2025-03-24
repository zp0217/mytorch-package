#calling packages
import os
import numpy as np
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

#define class load_data
class load_data:
    #function loads data(csv,json)
    def tabular_data(file_path):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path) #read csv
        elif file_path.endswith('.json'):
            return pd.read_json(file_path) #read json file
    
    #function that loads image
    def load_images(file_path):
        if file_path.endswith('.npz'):
            try:
                with np.load(file_path) as data:
                    return dict(data)  # Returns a dictionary of numpy arrays
            except Exception as e:
                print(f"Error loading npz file: {e}")
                return None
        else:
            images = []
            try:
                for filename in os.listdir(file_path):
                    img_path = os.path.join(file_path, filename)
                    img = Image.open(img_path).convert('RGB')
                    images.append((filename, img))
            except Exception as e:
                print(f"Error loading images from directory: {e}")
            return images
    #function to process image
    def preprocess_image(image, size=(128, 128), normalize=True):
        image = image.resize(size) #re-sizing image
        #normalizing image
        image_array = np.array(image, dtype=np.float32) / 255.0 if normalize else np.array(image)
        return image_array
    #function to load text data
    def text_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_list = [list(map(float, line.split())) for line in lines]
        return np.array(data_list)
    #function for text processing for text data
    def preprocess_text(text_lines, one_hot_encode=False):
     # Removes special characters and converts to lowercase
     cleaned_text = [re.sub(r'[^a-zA-Z0-9 ]', '', line.lower()) for line in text_lines]
     if one_hot_encode:
        encoder = OneHotEncoder(sparse_output =False)
        reshaped_text = np.array(cleaned_text).reshape(-1, 1)
        return encoder.fit_transform(reshaped_text)
     return cleaned_text



    #function for split test,train,validation
    def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
     #split training and test data
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
     #split training data to training and validation set
     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
     return X_train, X_val, X_test, y_train, y_val, y_test
    #function to save data
    def save_data(data, file_path):
        ext = os.path.splitext(file_path)[-1]
        if ext == '.csv':
            data.to_csv(file_path, index=False) # save data as csv
        elif ext == '.json':
            data.to_json(file_path, orient='records', indent=4) #save data as json
