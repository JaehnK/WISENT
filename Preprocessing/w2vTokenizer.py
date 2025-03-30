
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer, BertModel
import torch
from gensim.models import Word2Vec
import nltk
from collections import defaultdict
