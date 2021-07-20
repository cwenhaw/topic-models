# NMF topic models

**Topic modeling with Non-negative matrix factorization**

Minimizes the following loss function using multiplicative updates:

<img src="https://render.githubusercontent.com/render/math?math=||X-WH||^{2}_{Fro} %2B \alpha(||W||_1%2B||H||_1)">

Let D be no. of documents and V be the vocab size. **X** is (D x V) data matrix. Each row is a document and each column is a feature, e.g. textual/visual word. **W** is document-topic matrix of dimension (D x K) where K is the number of topics. **H** is topic-word matrix of dimension (K x V). 

