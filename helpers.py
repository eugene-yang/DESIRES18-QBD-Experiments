import argparse

import numpy as np
import pandas as pd
from pathlib import Path

import xml.etree.ElementTree as ET

from progressbar import ProgressBar, Bar, Percentage, ETA, Counter

def pbar(m):
    return ProgressBar(widgets=[Counter(), "  ", Bar(), Percentage(), ' (', ETA(), ')'], 
                       max_value=m)

def getText(fn):
	root = ET.parse( str(fn) )
	title = root.find("title").text
	headline = root.find("headline").text
	dateline = root.find("dateline").text if root.find("dateline") is not None else ""
	text = (" ".join([ c.text for c in root.find("text") ])).replace("--", "")
	return "\n".join(map(lambda x: "" if x is None else x, [title, headline, dateline, text]))

def ingestCollection(data_path, v2dids, verbose, **kwargs):
    """Ingest the collection and return the data

    Parameters
    ----------
    data_path : str | pathlib.Path
        path to the RCV1 document directory
    
    v2dids : list
        list of RCV1-v2 dids
    
    verbose : bool
        Whether to go verbose
    
    Retruns
    -------
    raw_text : pandas.DataFrame
        DataFrame containing all text from each document

    """
    if not isinstance( data_path, Path ):
        data_path = Path( data_path )
    fns = list( data_path.glob("*/*.xml") )
    if verbose:
        fns = pbar( len(fns) )(fns)
    return pd.DataFrame([
        { "filepath": str( fn.relative_to(data_path) ),
          "did": int(fn.name.replace("newsML.xml", "")),
          "raw": getText(fn) } for fn in fns
    ]).sort_values('did')

def parseQrels(data_path, raw_text, rel_threshold=26, verbose=False, **kwargs):
    """Parse the .qrels files and retern the DataFrame containing the relevant mask
    according to the dids containing in raw_text

    Parameters
    ----------
    data_path : str | pathlib.Path object
        directory containing all 3 .qrels files
    
    raw_text : pandas.DataFrame
        ingested collection dataframe
    
    rel_threshold : int, default 26
        the minimum threshold of relevant document in a category,
        only for experiment purpose

    verbose : bool
        Whether to go verbose

    Returns
    -------
    rel_info : pandas.DataFrame
        pandas DataFrame object containing all the relevant information

    """
    if not isinstance( data_path, Path ):
        data_path = Path( data_path )

    if verbose: print("Parsing .qrels files...")
    qrels = {
        f.name.split(".")[1]: 
            pd.DataFrame([ l.strip().split() for l in f.open() ], 
                         columns=['cate', 'did', 'rel'])
        for f in data_path.glob("*.qrels")
    }
    qrels = pd.concat([ 
        qrels[k].assign( cate=qrels[k].cate.map(lambda x:k+"_"+x) ) for k in qrels ])

    # should only contains dids in RCV1-v2
    dids = raw_text['did']

    if verbose: print("Creating masks...")
    rel_info = {
        cate: np.isin( dids, df.did.astype(int) )
        for cate, df in qrels.groupby('cate')
    }

    return pd.DataFrame({ cate: rel_info[cate] 
                for cate in rel_info if rel_info[cate].sum() >= rel_threshold })\
           .assign( did = dids )

def sampleQuery(rel_info, num_sampled = 25):
    return pd.DataFrame({
        cate: np.random.permutation( rel_info.index[ rel_info[cate] ] )[:num_sampled]
        for cate in rel_info.drop('did', axis=1)
    })

def args():
    parser = argparse.ArgumentParser(description='Helper functions')
    sps = parser.add_subparsers(dest='cmd')
    
    parser_ingest = sps.add_parser('ingest', 
                                   help='Ingest the raw RCV1-v2 dataset '
                                        'and output a .csv file.')
    parser_ingest.add_argument('data_path', type=str,
                               help='Path the the root directory of the dataset.')
    parser_ingest.add_argument('output_path', type=str, help='Output file name (.csv)')
    parser_ingest.add_argument('--v2_did_list', type=str,
                               default='RCV1-v2/v2dids.txt',
                               help='File path to the .txt file containing the '
                                    'did of the v2 collection. This file is already '
                                    'included in the repository.')
    
    parser_qrels = sps.add_parser('qrels', help='Parse .qrels file and create binary '
                                                'mask file(.pkl) of relevant judgements.')
    parser_qrels.add_argument('data_path', type=str, 
                              help='Path to the directory containing all 3 .qrel files')
    parser_qrels.add_argument('output_path', type=str, help='Output file name (.pkl)')
    parser_qrels.add_argument('raw_text_file', type=str, 
                              help='The ingested .csv file created by `helper.py ingest`')
    parser_qrels.add_argument('--rel_threshold', type=int, default=26,
                              help='Minimum number of relevant document of a category')
    
    parser_samp = sps.add_parser('sample', 
                                 help='Sample from the collection to create query document '
                                      'of each category.')
    parser_samp.add_argument('rel_info_file', type=str,
                             help='The relevant mask file created by `helper.py qrels`')
    parser_samp.add_argument('output_path', type=str, help='Output file name (.pkl)')
    parser_samp.add_argument('--num_sampled', type=int, default=25,
                             help='Number of query document per category.')

    parser.add_argument('--silent', action='store_true', default=False,
                        help='Flag for not output status information')

    return parser.parse_args()

if __name__ == '__main__':
    args = args()
    if args.cmd == 'qrels':
        raw_text = pd.read_csv( args.raw_text_file )
        ret = parseQrels( **vars(args), raw_text=raw_text, 
                          verbose=not(args.silent) )
        if not args.silent: print("Saving...")
        ret.to_pickle( args.output_path )

    elif args.cmd == 'ingest':
        v2dids = [ int(l) for l in open( args.v2_did_list ) ]
        raw_text = ingestCollection( args.data_path, v2dids,
                                     verbose=not(args.silent) )
        if not args.silent: print("Saving...")
        raw_text.to_csv( args.output_path )
    
    elif args.cmd == 'sample':
        rel_info = pd.read_pickle( args.rel_info_file )
        queries = sampleQuery( rel_info, args.num_sampled )
        if not args.silent: print("Saving...")
        queries.to_pickle( args.output_path )

    else:
        print("418 I'm a teapot")