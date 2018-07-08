import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sp

import json, warnings, socket, pickle, sys, re

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from helpers import pbar

parser = argparse.ArgumentParser(description='Dump Elasticsearch index as a document-term '
									   'sparse matrix that can be used in `sklearn_exp.py`')
parser.add_argument('server', type=str, help='Path to the Elasticsearch server, including '
											 'hostname/IP and port')
parser.add_argument('index', type=str, help='Name of the Elasticsearch index to be dumped.')
parser.add_argument('output_name', type=str, help='Output file name (.pkl)')

parser.add_argument('--ingested_text', type=str, default="./raw_text.csv",
					help='The ingested .csv file created by `helper.py ingest`')

args = parser.parse_args()

if __name__ == '__main__':
	es = Elasticsearch([ args.server ])

	print("Loading text cache...")
	raw_text_index = pd.read_csv( args.ingested_text ).index

	finalfn = Path( args.output_name + ".pkl" )

	if finalfn.exists():
		raise FileExistsError("%s already exists."%finalfn)

	print("Start dumping from Elasticsearch")
	doc_terms = []
	vocab = set()

	for i in pbar(raw_text_index.shape[0])(raw_text_index):
		d = es.termvectors(index=args.index, doc_type='document', id=i, fields='raw')
		vocab |= set( d['term_vectors']['raw']['terms'].keys() )
		doc_terms.append({ v: d['term_vectors']['raw']['terms'][v]['term_freq'] 
							for v in d['term_vectors']['raw']['terms'] })

	vocab = sorted(list(vocab))

	print("Vectorizing...")
	n_samples = raw_text_index.shape[0]
	n_features = len( vocab )
	inv_vocab = { v:i for i,v in enumerate(vocab) }

	row = []
	col = []
	data = []
	for i, doc in pbar(n_samples)(enumerate( doc_terms )):
		for term in doc:
			row.append(i)
			col.append( inv_vocab[term] )
			data.append( doc[term] )

	print("Transforming...")
	X = sp.csr_matrix(( data, (row,col) ), shape=(n_samples, n_features))

	print("Saving...")
	pickle.dump({ "vec": X }, finalfn.open("wb") )





