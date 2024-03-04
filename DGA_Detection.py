import json

import tldextract
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Removed unused import statement
# import tensorflow as tf

# load model and config
conf = json.load(open("./config.json"))
valid_chars = conf["valid_chars"]
max_len = conf["max_len"]
max_feature = conf["max_feature"]
custom_objects = {
    "Embedding": Embedding(max_feature, 128, input_length=max_len),
    "LSTM": LSTM(128),
    "Dense": Dense(1),
    "Dropout": Dropout(0.5),
    "Activation": Activation("sigmoid"),
}
model = load_model("./model/dga_detection.v1.keras")


def predict(domains, threshold=0.5):
    domain = [
        [valid_chars[ch] for ch in tldextract.extract(domain).domain]
        for domain in domains
    ]
    domain = pad_sequences(domain, maxlen=max_len)

    predicted = model.predict(domain)

    return (predicted > threshold).astype(int)


print(predict(["google.com", "google.com", "google.com"]))
