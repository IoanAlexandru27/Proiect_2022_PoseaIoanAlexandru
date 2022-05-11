from gensim.models import Word2Vec
import pandas
import sys
import numpy as np
import nltk
import re
import itertools
from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import FOAF, RDF
import urllib.parse

tastes = [
    "toast",
    "cigar",
    "cherry",
    "coffee"
]

varieties = ["Pinot Noir",  "Tinta de Toro", "Sauvignon Blanc"]

evil_words = [
"and", "the", "with", "a", "of", "wine", "this", "is", "flavors", "that", "on", "aromas"
]

def GetRankDiff(model, familyword, word2test):
    try:
        x = np.array(model.wv[familyword] - model.wv[word2test])
        return np.linalg.norm(x)
    except:
        return 100

def CountWords(lst):
    dic = {}

    for word in lst:
        if word not in evil_words:
            if word not in dic:
                dic[word] = 1
            dic[word] += 1

    res = []
    for k in sorted(dic, key=dic.get, reverse=True):
        res.append((k, dic[k]))
    return res[:10]


def SaveFile(name, data, dataname):
    print(zip(data, dataname))
    g = Graph()
    g.bind("foaf", FOAF)
    wine = URIRef("http://tema/wine")
    tastesLike = URIRef("http://tema/tastesLike")
    hasVariety = URIRef("http://tema/hasVariety")
    for d, dn in zip(data, dataname):
        wineType = URIRef("http://tema/" + urllib.parse.quote(dn))
        g.add((wine, hasVariety, wineType))
        for taste in d:
            tasteProp = Literal(taste[0])
            g.add((wineType, tastesLike, tasteProp))
    print(g.serialize())

        


def main():
    model = Word2Vec.load("Model.wv")
    data = pandas.read_csv("winemag-data_first150k.csv")

    result = {}
    for desc, name in zip(data["description"], data["variety"]):
        if name in varieties:
            if name not in result:
                result[name] = []
            for word, taste_sample in itertools.product(desc.split(), tastes):
                diff = GetRankDiff(model, "vanilla", word)
                if diff <= 2.6:
                    result[name].append(word)


    tasteLists = []
    for var in varieties:
        tasteLists.append(CountWords(result[var]))

    SaveFile("fd", tasteLists, varieties)

if __name__ == "__main__":
    main()
