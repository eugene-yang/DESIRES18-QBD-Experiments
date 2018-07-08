import argparse
from pathlib import Path
import pandas as pd
import pickle

from bm25_external import Bm25Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def args():
    parser = argparse.ArgumentParser(description='Generating vector file for sklearn experiments')
    parser.add_argument('ingested_text_file', type=str)
    parser.add_argument('output_file', type=str)

    sps = parser.add_subparsers(dest='style')
    parser_tfidf = sps.add_parser('tfidf')
    parser_tfidf.add_argument('--norm', choices=['l2', 'l1', 'None'], default='l2')
    parser_tfidf.add_argument('--sublinear_tf', action='store_true', default=False)
    parser_tfidf.add_argument('--use_idf', action='store_true', default=False)
    parser_tfidf.add_argument('--smooth_idf', action='store_true', default=False)

    parser_bm25 = sps.add_parser('bm25')
    parser_bm25.add_argument('--use_idf', action='store_true', default=False)
    parser_bm25.add_argument('--smooth_idf', action='store_true', default=False)
    parser_bm25.add_argument('--es_style_idf_smooth', action='store_true', default=False)
    parser_bm25.add_argument('--k', type=float, default=1.2)
    parser_bm25.add_argument('--b', type=float, default=0.75)

    args = parser.parse_args()

    if args.style == 'tfidf' and args.norm == 'None':
        args.norm = None

    return args

if __name__ == '__main__':
    args = args()
    print('Loading ingested text file...')
    raw_text = pd.read_csv( args.ingested_text_file )

    model_args = { k:v for k,v in vars(args).items() 
                    if k not in ['ingested_text_file', 'output_file', 'style'] }
    if args.style == 'tfidf':
        model = TfidfVectorizer(**model_args)
    else:
        model = Bm25Vectorizer(**model_args)
    
    print('Transforming...')
    pickle.dump({ "vec": model.fit_transform( raw_text.raw.tolist() ) }, open( args.output_file,  "wb") )
