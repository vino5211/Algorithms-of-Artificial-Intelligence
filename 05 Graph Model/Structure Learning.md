## Define of Structure Learing
+ Input and output are both objects with structures
+ objects : sequence, list, tree, bounding box, **not vectors**
+ find function f 
$$
	f : X \rightarrow Y
$$
+ Example Application
	+ Speech recognition
        + X : speech signal(sequence) $\rightarrow$ Y : text(sequence)
    + Translation
        + X : text(sequence) $\rightarrow$ Y : text(sequence)
    + Syntactic Parsing
        + X : sentence $\rightarrow$  Y : parsing tree(tree structure)
    + Object Detection
        + X : image $\rightarrow$ Y : bounding box
    + Summarization
        + X : long ducument $\rightarrow$ summary (short paragraph)
    + Retrieval
        + X : keyword $\rightarrow$ Y : search result(a list of webpage)

## Basic question of Structure Learning
+ Define
    + What does $ F(X,Y) $ look like ?
    + Also can be considered as probability $P(X,Y)$
        + MRF
        + Belief
        + Energy base model
+ Training
    + Find a function F, input is X, Y, output is a real number(R)
    $$ F : X * Y \rightarrow R$$
+ Inference
    + Given an object x, find $ \hat y $ as follow
    $$ \hat y = arg\ \underset{y \epsilon Y }{max} \  F(X,Y)$$ 
+ Detail example of Object Detection