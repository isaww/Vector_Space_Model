import numpy as np
import math
import os

class VSM():
    def __init__(self):
        self._words = {}  # map words to vocab index
        self._words2N = {} # number of files containing the word
        self._words2doc = {}  # word index to doc ID
        self._vectors = None  # 2D matrix, row: doc's tf-idf vector
        self._idf = None      # 1D vector
        self._docIDs = {}  # doc ids
        self._docnames = {}  # doc_ID to doc_name
        self._wordcount = 0   # len(vocab.all)
        self._doccount = 0  # len(file-list)
        self._sparsity = 0  # non-zero entries in _vectors
        self._vectors2 = None


    def read_inverted(self, f_path):
        with open(f_path) as f:
            for line in f:
                line = line.replace('\n','')
                _line = line.split(' ')
                if len(_line) == 3:
                    _wid1 = int(_line[0])
                    _wid2 = int(_line[1])
                    _count = int(_line[2])
                    self._words2N[_wid1] = _count
                    self._words2doc[_wid1] = {}
                else:
                    _did = int(_line[0])
                    _did_count = int(_line[1])
                    self._words2doc[_wid1][_did] = _did_count
        #self._wordcount = len(self._words2N)
        #print ('# of words: ', len(self._words2N.keys()))  # 22220
        #print ('word "1": ', self._words2N['1'])           # 2
        #print ('word "1": ', self._words2doc['1'])         #  {'33689': 1, '38365': 1}

    def vocab_to_index(self, f_path):
        with open(f_path) as f:
            _count = 0
            for line in f:
                line = line.replace('\n','')
                self._words[line] = int(_count)
                _count += 1
        self._wordcount = _count
        #print (self._wordcount)  # 29908

    def doc_to_index(self, f_path):
        with open(f_path) as f:
            _count = 0
            for line in f:
                line = line.replace('\n','')
                self._docIDs[line] = int(_count)
                #self._docnames[ int(_count) ] = line
                _count += 1
        self._doccount = _count
        #print (self._doccount)  # 46972

    def parse_xml(self, f_path):
        import xml.etree.ElementTree as ET
        root = ET.parse(f_path).getroot()
        for r2 in root:
            for child in r2:
                if child.tag == 'id':
                    _id = child.text
                elif child.tag == 'title':
                    _title = child.text
                elif child.tag == 'text':
                    _text = ''
                    for child2 in child:
                        _text += child2.text
                    _text = _text.replace('\n','')
        return _id, _title, _text


    def process_docs(self, f_path):
        # Initialize vector: len(doc) * len(vocab)
        self._vectors = np.zeros((self._doccount,self._wordcount))
        self._idf = np.zeros((self._wordcount))

        # Iterate through docs, adding docs to _vectors
        for subdir, dirs, files in os.walk(f_path):
            for file in files:
                f_name = os.path.join( '/'.join(subdir.split('/')[-3:]) , file)
                f_ind = self._docIDs[ f_name ]
                self._docnames[ f_ind ] = file
                
                f_name_full = os.path.join(subdir, file)
                d_id, d_title, d_text = VSM.parse_xml(self, f_name_full)

                d_text = [x for x in d_text if x in self._words.keys()]
                for w in d_text:
                    try:
                        self._vectors[ f_ind ][ self._words[w] ] += 1
                    except KeyError: pass
                
                
        # Calculate idf: log(n/df)
        __N = self._doccount
        for w in self._words2N.keys():           
            __df = self._words2N[w]
            self._idf[ w ] = np.log( (__N - __df + 0.5)/ (__df + 0.5) )
            #self._idf[ w ] = np.log( __N/ __df )
        # Combine tf-idf: (1 + log(tf)) * (idf) 
        for i in range(len(self._vectors)):
            for j in range(self._wordcount):
                if self._vectors[i][j] != 0:
                    ## logarithm
                    self._vectors[i][j] = (1 + np.log(self._vectors[i][j]) ) * self._idf[j]
                    ## Augmented
                    #_max_tf = 50000
                    #self._vectors[i][j] = (0.5 + (0.5*self._vectors[i][j]/_max_tf) ) * self._idf[j]
                    ## Boolean
                    #self._vectors[i][j] = 1
                    self._sparsity += 1
        # Normalize vector
        for i in range(len(self._vectors)):
            xx = np.linalg.norm(self._vectors[i])
            if xx != 0:
                self._vectors[i] = self._vectors[i] / xx

        
    def process_query(self, f_path, n_tags):
        # n_tags: list of contexts considered (e.g. 0-title, 1-question, 2-narrative, 3-concept)
        q_dict = {}  # q_dict['CIRB010TopicZH001'] = ['title','question','narrative','concept']
        q_count = 0
        q_names = []

        import xml.etree.ElementTree as ET
        root = ET.parse(f_path).getroot()
        for r2 in root:
            q_count += 1
            for child in r2:
                if child.tag == 'number':
                    _id = child.text
                    _id = _id.replace('\n','')
                    q_names.append(_id)
                elif child.tag == 'title':
                    _sen = child.text.replace('\n','')
                    q_dict[_id] = [_sen]
                elif child.tag == 'question':
                    _sen = child.text.replace('\n','')
                    q_dict[_id].append(_sen)
                elif child.tag == 'narrative':
                    _sen = child.text.replace('\n','')
                    q_dict[_id].append(_sen)
                elif child.tag == 'concepts':
                    _sen = child.text.replace('\n','')
                    q_dict[_id].append(_sen)
                
        q_vector = np.zeros((q_count, self._wordcount))
        for q_no in range(q_count):
            # q_vector[q_no], q_names[q_no], q_dict[ q_names[q_no] ]
            # E.g. n_tags = [0,1]
            q_text = q_dict[ q_names[q_no] ]  # list
            for tag in n_tags:
                for _char in q_text[tag]:
                    try:
                        q_vector[ q_no ][ self._words[_char] ] += 1
                    except KeyError: pass


        # Combine tf-idf: (1 + log(tf)) * (idf) 
        for i in range(len(q_vector)):
            for j in range(self._wordcount):
                if q_vector[i][j] != 0:
                    q_vector[i][j] = (1 + np.log(q_vector[i][j]) ) * self._idf[j]
        # Normalize vector
        for i in range(len(q_vector)):
            xx = np.linalg.norm(q_vector[i])
            if xx != 0:
                q_vector[i] = q_vector[i] / xx

        return q_dict, q_vector
    
    def get_cosine_similarity(self, x, y):
        # query, doc_vector: normalized vectors
        '''
        value = 0.
        xx = np.linalg.norm(x)
        yy = np.linalg.norm(y)
        xy = x.dot(y)
        if (xx*yy) != 0:
            return value/(xx*yy)
        else:
            return value
        '''
        return x.dot(y)
    
    def query_doc_sim(self, query, k):
        # for a given query, find relevant docs
        # query = q_vector[ i ]
        score = []
        for i in range(len(self._vectors)):
            #score.append( (-1 * (self.get_cosine_similarity(query, self._vectors[i])), i) )
            _s = self.get_cosine_similarity(query, self._vectors[i])
            score.append( (_s, i) )

        # Obtain top-k docs
        score.sort(key=lambda tup: tup[0], reverse=True)
        rel_docs = []
        for i in range(k):
            rel_docs.append(score[i][1])

        return rel_docs
    
    def do_rocchio(self, seed_docs):
        # for a seed doc, find relevant docs
        # for rocchio feedback
        
        # Get centroid
        avg_emb = np.zeros((self._wordcount))
        cnt = 0
        for i in range(len(seed_docs)):
            _w = np.log(len(seed_docs)+1-i)
            #_w = len(seed_docs)-i
            avg_emb += (self._vectors[seed_docs[i]] * _w)
            cnt += _w
        #avg_emb = avg_emb / float(len(seed_docs))
        avg_emb = avg_emb / cnt

        # Do filtering
        do_filtering = False
        if do_filtering==True:
            filter_score = []
            for i in range(len(seed_docs)):
                _s = self.get_cosine_similarity(avg_emb, self._vectors[seed_docs[i]])
                filter_score.append((seed_docs[i], _s))
            filter_score.sort(key=lambda tup: tup[1], reverse=True)
            # New centroid
            avg_emb = np.zeros((self._wordcount))
            cnt = 0
            seed_keep = []
            for i in range(len(seed_docs)-2):
                seed_keep.append(filter_score[i][0])
            for i in range(len(seed_docs)):
                if seed_docs[i] in seed_keep:
                    _w = np.log(len(seed_docs)+1-i)
                    avg_emb += (self._vectors[seed_docs[i]] * _w)
                    cnt += _w
            avg_emb = avg_emb / cnt


        score_seed = np.zeros((self._doccount))
        for j in range(len(self._vectors)):
            _s = self.get_cosine_similarity(avg_emb, self._vectors[j])
            score_seed[ j ] += _s

        return score_seed

    def do_rocchio_q(self, query):
        score_query = np.zeros((self._doccount))
        for j in range(len(self._vectors)):
            _s = self.get_cosine_similarity(query, self._vectors[j])
            score_query[ j ] += _s

        return score_query

    def do_searching(self, f_query, f_output, do_rocchio=False):
        with open(f_output, 'w') as f:
            f.write('query_id,retrieved_docs')
            f.write('\n')
            q_dict, q_vector = self.process_query(f_query, [0,1,3,3,3])
            #q_dict, q_vector = self.process_query(f_query, [0,3])
            q_names = list(q_dict.keys())
            for i in range(len(q_vector)):
                if do_rocchio==False:
                    rel_docs = self.query_doc_sim(q_vector[i], 100)
                else:   
                    # perform rocchio feedback
                    num_seeds = 10
                    alpha = 0.4  # query weight
                    beta = 0.6   # seed doc weight
                    seed_docs = self.query_doc_sim(q_vector[i], num_seeds)
                    score_seed = self.do_rocchio(seed_docs)  # np array
                    score_query = self.do_rocchio_q(q_vector[i])  # np array
                    score_exp = alpha*score_query + beta*score_seed
                    score_list = []
                    for s in range(len(score_exp)):
                        score_list.append( (score_exp[s], s) )
                    score_list.sort(key=lambda tup: tup[0], reverse=True)

                    rel_docs = []
                    #rel_docs = seed_docs
                    for s in range(100):
                        rel_docs.append(score_list[s][1])


                doc_names = [self._docnames[ rel_docs[d] ].lower() for d in range(len(rel_docs))]
                f.write(q_names[i][-2:])
                f.write(',')
                f.write(' '.join( doc_names ))
                f.write('\n')

#data/CIRB010/cdn/chi/cdn_chi_0000002
#CIRB010/cdn/loc/CDN_LOC_0001457 
    '''    
    def process_docs(self, docs):
        # docs: list(doc_ID, doc_content)
        for doc in docs:
            for word in doc[1]:
                if not word in self._words:
                    self._words[word] = self._wordcount
                    self._wordcount += 1
    '''

