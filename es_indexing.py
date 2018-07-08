import argparse
from pathlib import Path
import pandas as pd
import json

from elasticsearch import Elasticsearch

from helpers import pbar

def parseArgs():
	parser = argparse.ArgumentParser(description='Script for indexing the RCV1-v2 collection '
												 'in Elasticsearch')
	parser.add_argument('server', type=str, help='Path to the Elasticsearch server, including '
												 'hostname/IP and port')
	parser.add_argument('--raw_text', type=str, default="./raw_text.csv",
						help='Ingested .csv file created by `helper.py ingested`')
	parser.add_argument('--dataset_name', type=str, default="rcv1",
						help='The name of the collection, would be the prefix of the indices')
	parser.add_argument('--index_config', type=str, default="./es_config.json",
						help='The .json config file of the indices, already included in the '
							 'repository.')
	parser.add_argument('--es_bulk_max', type=int, default=1000,
						help='The maximum number of document passing to the server at once.')

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
	args = parseArgs()
	
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