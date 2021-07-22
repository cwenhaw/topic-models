import tomotopy as tp
#from dso_coherence import *
import nltk
from nltk.corpus import stopwords
import re
from jpp import JAL_NMF
import numpy as np
from scipy.sparse import csr_matrix

stopwords=stopwords.words('english')
punctuations=set([',','.','?','!','-', '_'])
with_num=False
with_punc=False

def get_plain(content):
    content_plain = re.sub(r'[^\x00-\x7f]',r'',content.strip())
    if with_num==False:
            content_plain = re.sub("[0-9]", " ", content_plain)  # strip numbers
    if with_punc==False:
            content_plain = re.sub(r'[^\w\s]',' ',content_plain) # strip puntuaction
    content_plain = re.sub("\s\s+" , " ", content_plain) # shrink multiple whitespace into single whitespace
    return content_plain



def preprocess():    
    fout=open('text.txt','w')
    fp=open('img_ocr.txt','r')
    for line in fp:
        #print(line)
        tokens=line.strip().split('\t')
        if len(tokens)>1:
            text=tokens[1].lower()
            text=get_plain(text)
            words=text.split(' ')
            for w in words:
                if len(w)<=1:
                    continue
                if (w not in stopwords) and (w not in punctuations):
                    fout.write(w+' ')
            fout.write('\n')
    fp.close()        
    fout.close()        

def getKey(item):
    return item[1]

def stats():
    wfreq=dict()
    fp=open('text.txt','r')
    for line in fp:
        tokens=line.strip().split(' ')
        for w in tokens:
            if w not in wfreq:
                wfreq[w]=0
            wfreq[w]+=1
    fp.close()
    temp=[]
    for w, freq in wfreq.items():
        temp.append([w,freq])
    sorted_list=sorted(temp, key=getKey, reverse=True)        
    fp=open('vocab.txt','w')
    for w, freq in sorted_list:
        fp.write(w+' '+str(freq)+'\n')
    fp.close()        

# memes about same topic can appear visually very different, e.g. 'covid'
def fit_LDA():
    mdl = tp.LDAModel(k=50, seed=123, min_cf=3, min_df=3, rm_top=30)
    for line in open('text.txt'):
        mdl.add_doc(line.strip().split())

    mdl.train(500)
    topw=30
    topics=[]
    for k in range(mdl.k):
        #print(mdl.get_topic_words(k, top_n=topw))
        topn=[]
        for word, p in mdl.get_topic_words(k, top_n=topw):
            topn.append(word)
        print(topn)        

def form_X():
    wordmap=dict()
    rev_wordmap=dict()
    documents=[]
    wid=0
    fp=open('text.txt','r')
    for line in fp:
        tokens=line.strip().split(' ')
        doc=[]
        for w in tokens:
            if w not in wordmap:
                wordmap[w]=wid
                rev_wordmap[wid]=w
                wid+=1
            doc.append(wordmap[w])                
        documents.append(doc)                
    fp.close()

    row=[]
    col=[]
    data=[]
    docid=0
    for doc in documents:
        tf=dict()
        for w in doc:
            if w not in tf:
                tf[w]=0
            tf[w]+=1
        for w, freq in tf.items():
            row.append(docid)
            col.append(w)
            data.append(freq)
        docid+=1            

    X=csr_matrix((data, (row, col)), shape=(len(documents), len(wordmap)))
    print(X.shape)
    print(X.count_nonzero())
    return [X, wordmap, rev_wordmap]




def NMF_expt():
    maxIter=100
    #N=13000
    #V=26000    
    #X=np.random.rand(N,V)
    K=10
    [X, wordmap, rev_wordmap]=form_X()
    computeLoss=True
    l1reg=0.10   # L1 regularization parameter
    topn=10
    for run in range(0, 1):
        np.random.seed(run)
        [W, H]=JAL_NMF(X, K, l1reg, maxIter, computeLoss)
        print(np.sum(W))
        print(np.sum(H))

        for k in range(0, K):
            sorted_wid=H[k,:].argsort()[::-1]
            top_words=[]
            for wid in sorted_wid[:topn]:
                top_words.append(rev_wordmap[wid])
            print(top_words)



'''
preprocess()
stats()
'''
#fit_LDA()
NMF_expt()

exit(0)

import cv2 as cv
img = cv.imread('test.jpg')
#surf = cv.xfeatures2d.SURF_create(400)     # patented, need licence
#kp, des = surf.detectAndCompute(img,None)



# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None)
print(len(kp1))
print(des1.shape)
print(des1[0,])
