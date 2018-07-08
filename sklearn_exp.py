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
parser.add_argument('topic', type=str, help='Experimenting category')
parser.add_argument('feature_file', type=str, 
					help='The collection document representation. '
					     'Vector file created by `vectorize.py`')

parser.add_argument('style', choices=['similarity', 'onehot', 'other'], default='similarity',
					help='Style of querying, aka the query document representation'
						 '`Similarity` would make the query document having the same '
						 'representation as the collection documents; '
						 '`Onehot` would make query document having one hot encoding of the terms; '
						 '`Other` would require to specified the alternative vector file')
parser.add_argument('--query_feature_file', type=str, default='',
					help='Required if `--style=other`. The alternative representation of the '
						  'query documents. Vector file created by `vectorize.py`')
parser.add_argument('--rel_info_file', type=str, default='rel_info.pkl',
					help='Relevant binary mask .pkl file created by `helper.py qrels`')
parser.add_argument('--sampled_query_file', type=str, default='sampled_queries.pkl',
					help='Query document .pkl file created by `helper.py sample`')

parser.add_argument('--trials', type=int, default=25,
					help='Number of query document being experimented. Should be the same as'
						 'the number of query documenty in `sample_query_file`.')
parser.add_argument('--out_dir', type=str, default='./results',
					help='Output file directory. Will be created if not exist')
parser.add_argument('--prefix', type=str, default='',
					help='The prefix of this experiment run.')
parser.add_argument('--exp_name', type=str, default='sklearn_exp',
					help='The name of the experiment.')

parser.add_argument('--residual', action='store_true', default=False,
					help='The flag for using residual effectiveness. Please always set it '
						 'to True!')
parser.add_argument('--overwrite', action='store_true', default=False,
					help='Overwriting the existed experiment')

args = parser.parse_args()

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

	json.dump( {**vars(args), "machine": socket.gethostname(), "script": __file__}, 
			   (output_dir / "arguments.json").open("w") )
	
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

		Y_pseudo = np.zeros( Y.shape[0] )
		Y_pseudo[ query ] = 1

		model_time = time.time()
		if args.style == 'similarity':
			sim = np.asarray( (X * X[query].transpose()).todense() ).transpose()[0]
		elif args.style == 'onehot':
			sim = np.asarray( (X * (X[query] > 0).astype(int).transpose()).todense() ).transpose()[0]
		else: # other
			sim = np.asarray( (X * X_query[query].transpose()).todense() ).transpose()[0]
		
		rank = pd.DataFrame({"sim": sim, "hashed": hashed_index})\
				 .sort_values(["sim", "hashed"], ascending=False).index
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

