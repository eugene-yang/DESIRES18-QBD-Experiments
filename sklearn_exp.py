import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
import math

from scipy import sparse

import errno, os, warnings, sys, time, socket, re
from pathlib import Path
import pickle, json
import argparse

from utils import *

__all_algos__ = ['lr', 'similarity', 'query', 'qtfraw', 'qbm25tf', 'qlogtf']
__all_weighings__ = ['size_inverse', 'noweights', 'noblind', 'noblindwidf', 'na']

parser = argparse.ArgumentParser(description='Sklearn experiments')
parser.add_argument('dataset', type=str)
parser.add_argument('topic', type=str)
parser.add_argument('feature_file', type=str)

parser.add_argument('--style', choices=['similarity', 'onehot', 'other'], default='similarity')
parser.add_argument('--query_feature_file', type=str, default='')
parser.add_argument('--rel_info_file', type=str, default='rel_info.pkl')
parser.add_argument('--sampled_query_file', type=str, default='sampled_queries.pkl')

parser.add_argument('--trials', type=int, default=25)
parser.add_argument('--out_dir', type=str, default='./results')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--exp_name', type=str, default='sklearn_exp')

parser.add_argument('--residual', action='store_true', default=False)
parser.add_argument('--overwrite', action='store_true', default=False)

args = parser.parse_args()


dataset_dir = Path(args.dataset)

output_name = ( args.prefix + ("_" if args.prefix != "" else "") + args.topic )

output_dir = Path(args.out_dir) / args.exp_name / args.feature / args.algo / output_name
print(output_dir)

if len(list( (output_dir.parent.glob("*%s"%(args.topic))) )) > 0:
	if args.overwrite:
		warnings.warn("Experiment already exists, may overwrite")
	else:
		raise Exception("experiment already exists")
for pa in reversed((output_dir/"t").parents):
	try: 
		pa.mkdir()
	except FileExistsError as e:
		pass


if args.trials > 25:
	raise Exception("Over maximum 25 trials")


def metrices(rank, Y, q):
	nrel = Y.sum()

	if args.residual:
		rank = rank[ rank != q ]
		nrel = nrel - 1

	res = { "r-precision": Y[ rank[0:nrel] ].sum() / nrel, "query": q }
	for p in [1, 5, 10, 20, 24, 25]:
		res[ "precision@%d"%p ] = Y[ rank[0:p] ].sum() / p

	return res
	


def main():
	sys.stdout = logger( output_dir / "stdout.log" )
	sys.stderr = logger( output_dir / "stderr.log" )

	print("[exp] Running on %s"%args.topic)

	json.dump( {**vars(args), "machine": socket.gethostname(), "script": __file__}, (output_dir / "arguments.json").open("w") )
	
	queries = pd.read_pickle( args.sampled_query_file )[ args.topic ].tolist()
	X = pd.read_pickle( args.feature_file )[ 'vec' ]
	Y = pd.read_pickle( args.rel_info_file )[ args.topic ]

	hashed_index = Y.index.map(lambda x:hash(str(x)))

	if args.style == 'other':
		X_query = pd.read_pickle( args.query_feature_file  )[ 'vec' ]


	meta = {}

	meta['queries'] = queries

	sims = []
	meta['model_params'] = []
	meta['measure'] = []
	meta['time'] = []

	for query in queries[:args.trials]:
		print("[exp] %s set #%d as query"%(args.topic, query))

		# Q = X[query]
		Y_pseudo = np.zeros( Y.shape[0] )
		Y_pseudo[ query ] = 1
		
		# use logistic regression

		model_time = time.time()
		
		if args.style == 'similarity':
			sim = np.asarray( (X * X[query].transpose()).todense() ).transpose()[0]
		elif args.style == 'onehot':
			sim = np.asarray( (X * (X[query] > 0).astype(int).transpose()).todense() ).transpose()[0]
		else: # other
			sim = np.asarray( (X * X_query[query].transpose()).todense() ).transpose()[0]
		

		rank = pd.DataFrame({"sim": sim, "hashed": hashed_index}).sort_values(["sim", "hashed"], ascending=False).index
		# rank = np.argsort(sim)[::-1]
		meta['time'].append( time.time() - model_time )
		sims.append(sim)
		met = metrices(rank, Y, query)

		print("[exp] similarity metrics: %s"%( json.dumps(met) ))
		meta['measure'].append( met )


	print("[exp] saving similarities and metadata for %s"%args.topic)
	np.save( str( output_dir / "sims" ), np.stack(sims) )
	
	json.dump( meta, ( output_dir / "meta.json" ).open("w") )

	print("[exp] done")


if __name__ == '__main__':
	main()

