import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import json, warnings, socket, pickle, sys, re
from traceback import print_exception
from multiprocessing import Pool
from functools import partial

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from utils import logger

parser = argparse.ArgumentParser(description='Elasticserach experiment')
parser.add_argument('server', type=str, help='Path to the Elasticsearch server, including '
											 'hostname/IP and port')
parser.add_argument('index', type=str, help='Name of index')
parser.add_argument('type', choices=['mlt', 'sqs'], 
					help='Type Elasticsearch query going to experiment. '
						 'MLT query / SQS query')

# parser.add_argument('--data_path', type=str, default="./raw_v2_dataset")
parser.add_argument('--ingestes_text', type=str, default="raw_text.csv",
					help='Ingested .csv file created by `helper.py ingested`')
parser.add_argument('--query_file', type=str, default="sampled_queries.pkl",
					help='Query document .pkl file created by `helper.py sample`')
parser.add_argument('--rel_file', type=str, default="rel_info.pkl",
					help='Relevant binary mask .pkl file created by `helper.py qrels`')
parser.add_argument('--output_dir', type=str, default='./results_es',
					help='Output file directory. Will be created if not exist')
parser.add_argument('--exp_name', type=str, default='es_mltvssqs',
					help='The name of the experiment.')
parser.add_argument('--type_prefix', type=str, default='',
					help='The prefix of this experiment run.')
parser.add_argument('--worker', type=int, default=4,
					help='Number of multiprocess workers')

parser.add_argument('--overwrite', action="store_true", default=False,
					help='Overwriting existed experiment, default False')
parser.add_argument('--skip_exist_cate', action="store_true", default=False,
					help='Skipping existed categories in this experiment run, '
						 'default False.')

parser.add_argument('--only_cate', type=str, default=None,
					help='Run only this category')

args = parser.parse_args()

server = socket.gethostname().split(".")[0]
output_dir = Path( args.output_dir ) / args.exp_name / \
			( ("" if args.type_prefix == '' else args.type_prefix + "_" ) \
			  + args.type + "_" + args.index)

if output_dir.exists():
	if args.overwrite:
		warnings.warn("Experiment already exists, may overwrite")
	elif args.skip_exist_cate or args.only_cate is not None:
		warnings.warn("Resuming experiment")
	else:
		raise Exception("experiment already exists")

for pa in reversed((output_dir/"t").parents):
	try: pa.mkdir()
	except FileExistsError: pass

def metrices(rank, Y, q):
	nrel = Y.sum() - 1
	rank = rank[ rank != q ]
	
	res = { "r-precision": Y[ rank[0:nrel] ].sum() / nrel, "query": q }
	for p in [1, 5, 10, 20, 24, 25]:
		res[ "precision@%d"%p ] = Y[ rank[0:p] ].sum() / p

	return res

def work(hashed, info):
	es = Elasticsearch([ args.server ])
	cate, qs, rel = info

	results = []
	rank_lists = []
	print("%s starts"%cate)

	nrel = rel.sum()
	try:
		for i,(qi, qt) in enumerate(qs.iteritems()):
			print("%s running query %d: #%d"%(cate, i, qi))
			# reg is important
			qt = re.sub(r'\W', ' ', qt)
			if args.type == 'mlt':
				esq = {
					"stored_fields": [],
					"query": {
						"more_like_this": {
							"fields": ["raw"],
							"like": qt
						}
					}
				}
			elif args.type == 'sqs':
				esq = {
					"stored_fields": [],
					"query": {
						"simple_query_string": { 
							"fields": ["raw"],
							"query": qt
						}
					}
				}
			
			resp = es.search(index=args.index, body=esq, size=rel.sum(), request_timeout=600, scroll="5m")
			rank = np.array([ int(r["_id"]) for r in resp['hits']['hits'] ])
			scores = np.array([ r["_score"] for r in resp['hits']['hits'] ])

			if rank.shape[0] <= nrel:
				# filling the rank list
				not_used = ~np.in1d( hashed.index, rank )
				adding = hashed.index[ not_used ][ : nrel - rank.shape[0] ]
				rank = np.concatenate([ rank, adding ])
				scores = np.concatenate([ scores, np.zeros( adding.shape[0] ) ])
			else:
				print("retry %s query #%d"%(cate,qi))

			# use hashed id to break ties
			df = pd.DataFrame({ "list": rank, "score": scores }).set_index('list').assign(hashed=hashed).sort_values(['score', 'hashed'], ascending=False)
			rank_lists.append({"query": qi, "list": df.index, "score": df['score']})
			results.append( metrices(df.index, rel, qi) )

		pd.DataFrame(results).to_csv( output_dir / (cate + ".csv"), index=False )
		pickle.dump( rank_lists, ( output_dir / (cate + "_ranks.pkl") ).open("wb") )
		print("%s Done"%cate)

	except Exception:	
		# keep going even if exeception
		print("error on %s"%cate)
		print_exception( *sys.exc_info() )


def main():
	sys.stdout = logger( output_dir / "stdout.log" )
	sys.stderr = logger( output_dir / "stderr.log" )
	
	print("Reading dataset cache from %s..."%args.ingestes_text )
	raw_text = pd.read_csv( args.ingestes_text )

	rel_info = pd.read_pickle( args.rel_file )

	queries = pd.read_pickle( args.query_file )

	# for tie-breaking
	hashed_index = pd.Series( rel_info.index.map(lambda x:hash(str(x))) ).sort_values()

	pair = lambda cate: (cate, raw_text.raw[ queries[cate] ], rel_info[cate])
	work_ = partial(work, hashed_index)
	if args.only_cate is not None:
		exps = [ pair( args.only_cate ) ]
	elif args.skip_exist_cate:
		exps = [ pair( cate ) for cate in queries.keys() if not ( output_dir / (cate + ".csv") ).exists() ]
	else:
		exps = [ pair( cate ) for cate in queries.keys() ]

	with Pool(args.worker) as p:
		p.map(work_, exps)

	print("all done")

if __name__ == '__main__':
	main()