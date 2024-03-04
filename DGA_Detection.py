import json

import tldextract
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences


class DGADetection:
    # load model and config
    def __init__(
        self, config_file="./config.json", model_file="./model/dga_detection.v1.h5"
    ):
        conf = json.load(open(config_file))
        self.valid_chars = conf["valid_chars"]
        self.max_len = conf["max_len"]
        self.max_feature = conf["max_feature"]
        self.model = Sequential()
        self.model.add(Embedding(self.max_feature, 128, input_length=self.max_len))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model.load_weights(model_file)

    def predict(self, domains, threshold=0.5):
        domain = [
            [self.valid_chars[ch] for ch in tldextract.extract(domain).domain]
            for domain in domains
        ]
        domain = pad_sequences(domain, maxlen=self.max_len)

        predicted = self.model.predict(domain)

        return [int(x[0]) for x in (predicted > threshold).astype(int)]
