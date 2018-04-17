CRF Tools 

- Neural CRF Parsing
  - 
- CRF++
  - Parameters
    - -t  产生文本格式的模型
  - Pos template
        # Unigram
        U00:%x[-3,0]
        U01:%x[-2,0]
        U02:%x[-1,0]
        U03:%x[0,0]
        U04:%x[1,0]
        U05:%x[2,0]
        U06:%x[3,0]
        U07:%x[-1,0]/%x[0,0]
        U08:%x[0,0]/%x[1,0]
        # Bigram
        B
  - References 
    - http://lovecontry.iteye.com/blog/1251227
    - CRF Paper
      - http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf
    - FlexCRFs
      - http://flexcrfs.sourceforge.net/flexcrfs.pdf
  - Todo
    - Support semi-Markov CRF
    - Support piece-wise CRF
    - Provide useful C++/C API (Currently no APIs are available)
  - Reference
    - J. Lafferty, A. McCallum, and F. Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data, In Proc. of ICML, pp.282-289, 2001
    - F. Sha and F. Pereira. Shallow parsing with conditional random fields, In Proc. of HLT/NAACL 2003
    - NP chunking
    - CoNLL 2000 shared task: Chunking
