from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras import layers
import numpy as np
import MeCab
import pickle 
from keras.models import load_model


input_text = input()
input_text = input_text.replace('　',' ')
tagger = MeCab.Tagger("-Owakati")
input_text = tagger.parse(input_text)
input_text = [input_text]
input_text = np.asarray(input_text)


max_len = 1000
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle) 
input_text = tokenizer.texts_to_sequences(input_text)
input_text = pad_sequences(input_text, maxlen=max_len)

model = load_model("novel_dp_model.h5")
predict = model.predict(input_text)

pre_result = []

for pre in predict[0]:
    pre_result.append(pre)
#pre_result.append(predict[0][0]+predict[0][1])
#pre_result.append(predict[0][2]+predict[0][3])
#pre_result.append(predict[0][4]+predict[0][5])
#pre_result.append(predict[0][6])
#pre_result.append(predict[0][7])
#pre_result.append(predict[0][8])
#pre_result.append(predict[0][9])
#pre_result.append(predict[0][10])
#pre_result.append(predict[0][11]+predict[0][12]+predict[0][13]+predict[0][14])

print(pre_result)