# Retrieval and Richness when Querying by Document

*Paper will be appeared in DESIRES 2018*

The experiment framework start from the ingestion of RCV1-v2 collection and the relevancy information. The detail information of the collection can be found in [this site](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm).

Detail arguments are listed in each python script by calling `python {script}.py --help`. 

### Environment Requirement

1. python >= 3.6
2. Elasticsearch Server 6.2 (has not test on >6.2 but should work)
3. Other python packages listed in `requirments.txt`

### Preprocessing Collection

All experiments will make use of the output files from the following three commands.

1. Ingestion of the raw collection
   `python helpers.py ingest {directory to collection} {output filename}`
2. Parse .qrels file
   `python helpers.py qrels {directory containing all 3 .qrels files} {output filename} {output file from 1}`
3. Create query document by sampling from the relevant documents of each category
   `python helpers.py sample {output file from 2} {output filename}`

### Elasticsearch Experiment

1. Index the collection in Elasticsearch 
   `python es_indexing.py {Elasticsearch server} `
   detail arguments please call `python es_indexing.py --help`
2. Run Elasticsearch experiments
   `python es_exp.py {Elasticserach server} {index name} {type of experimenting query} `
   detail arguments please call `python es_exp.py --help`

### Scikit-learn Experiment

1. Create document-term vector file
   `python vectorize.py {ingested collection} {style: tfidf/bm25}`
   detail arguments please call `python vectorize.py {ingested collection} {style} --help`
2. Dump Elasticsearch index as vector file
   `python es_dumpvec.py {Elasticsearch server} {index name} {output file} `
3. Run scikit-learn experiments
   `python sklearn_exp.py {category} {document vector file} {query style: similarity/onehot/other}`
   detail arguments please call `python sklearn_exp.py --help`

# Reference

Please kindly cite the following paper

Eugene Yang, David D. Lewis, Ophir Frieder, David Grossman, and Roman Yurchak. 2018. Retrieval and Richness when Querying by Document. In *Proceedings of Design of Experimental Search & Information REtrieval Systems (DESIRES 2018)* 