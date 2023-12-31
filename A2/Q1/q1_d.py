import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import os
import re
import sys

sw = set(STOPWORDS)
ps = PorterStemmer()

train_data_path = sys.argv[1]
train_dataset = load_train_data(train_data_path)
model = MultinomialEventModel(2)
model.fit(train_dataset)

test_data_path = sys.argv[2]
(testdata_pos, testdata_neg) = load_test_data(test_data_path)
predcnt_pos = np.count_nonzero(model.predict(testdata_pos) == 1)
predcnt_neg = np.count_nonzero(model.predict(testdata_neg) == 0)

acc = (predcnt_pos+predcnt_neg)/(len(testdata_pos)+len(testdata_neg))
print(f"# positive reviews correctly predicted: {predcnt_pos}")
print(f"# negative reviews correctly predicted: {predcnt_neg}")
print(f"Accuracy: ({predcnt_pos}+{predcnt_neg})/({len(testdata_pos)}+{len(testdata_neg)}) = {acc:.3f}")

make_wordcloud(model.cond_param_freqs[1], green_color_func, "wc_pos.png")
make_wordcloud(model.cond_param_freqs[0], red_color_func, "wc_neg.png")