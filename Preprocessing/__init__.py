import spacy
import fasttext
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

from .docs import Docs
from .sentense import Sentence
from .word import Word
from .trie import TrieNode, WordTrie

__all__ = ['Docs', 'Sentence', 'Word', 'TrieNode', 'WordTrie']