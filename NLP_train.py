import pandas
import nltk
import re
import gensim.models.word2vec as w2v
import multiprocessing

data = pandas.read_csv("winemag-data_first150k.csv")
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

all_descriptions = ""

print("Loading data...")

for desc in data["description"]: # extragerea tuturor descrierilor
    all_descriptions += desc

print("Data loaded!")


print("Tokenizing...")
sentences2tok = tokenizer.tokenize(all_descriptions) # tokenizare

sentences = []
for sent in sentences2tok: #curatare date
	if len(sent) > 0:
		clean = re.sub("[^a-zA-Z]"," ", sent)
		sentences.append(clean.split())

print("Tokenized!")


print("Training...")
embbeding = w2v.Word2Vec(
    sg=1,
    seed=1000,
    workers=multiprocessing.cpu_count(),
    min_count=2,
    window=10,
    sample=0.001
)

embbeding.build_vocab(sentences)

embbeding.train(sentences, total_examples=embbeding.corpus_count, epochs=5)

print("Trained!")

embbeding.save("Model.wv")