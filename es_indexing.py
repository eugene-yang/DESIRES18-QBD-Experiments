import argparse
from pathlib import Path
import pandas as pd
import json

from elasticsearch import Elasticsearch

from helpers import pbar

def args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('server', type=str)
	parser.add_argument('raw_text', type=str, default="./raw_text.csv")
	parser.add_argument('--dataset_name', type=str, default="rcv1")
	parser.add_argument('--index_config', type=str, default="./es_config.json")
	parser.add_argument('--es_bulk_max', type=int, default=1000)

	return parser.parse_args()

def bulk_index_docs(es, name, gen, bulk_max):
	opt = []
	for ind, content in gen:
		opt.append({'create': { '_index': name, '_type': 'document', '_id' : ind}})
		opt.append(dict(content))

		if len(opt) >= 2*bulk_max:
			es.bulk(body=opt, request_timeout=600)
			opt = []

	if len(opt) > 0:
		es.bulk(body=opt)

# vectorize all essential combinations

def main():
	args = args()
	
	print("Reading dataset cache from %s..."%args.raw_text)
	raw_text = pd.read_csv( args.raw_text )
	

	es = Elasticsearch([ args.server ])
	# create index
	index_config = json.load( open(args.index_config) )
	
	for sim in ["tfidf", "bm25", "lmds"]:
		name =  args.dataset_name + "_" + sim

		config = index_config
		config["mappings"]["document"]["properties"]["raw"]["similarity"] = ( "my_" + sim ) if sim != "tfidf" else "classic"

		print("create index with name: %s"%name)
		if es.indices.exists(index=name):
			es.indices.delete(index=name)
		es.indices.create(index=name, body=config)
		bar = pbar(raw_text.shape[0])
		bulk_index_docs(es, name, bar(raw_text.iterrows()), bulk_max = args.es_bulk_max )
	
if __name__ == '__main__':
	main()