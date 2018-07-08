import pickle, json
from .logger import logger

__all__ = ["logger", "aslist", "load_pickle", "load_json"]

load_pickle = lambda x: pickle.load( open( str(x), "rb" ) )
load_json = lambda x: json.load( open( str(x) ) )

def aslist(l):
	return l if isinstance(l, list) else [l]
