# tmproj
Text Mining Project


```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -preload tokenize,ssplit,pos,lemma,parse,depparse \
    -status_port 9000 -port 9000 -timeout 15000
```