# tmproj
SMU IS450 Text Mining Project
- Telegram bot that answers law-related questions

## Running the telegram bot

### installing prerequisites
```
pip install -r requirements.txt
```

### Run stanford parser onport 9000
```
cd stanford-corenlp-full-2016-10-31
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -preload tokenize,ssplit,pos,lemma,parse,depparse \
    -status_port 9000 -port 9000 -timeout 15000
```
- *** the telegram bot needs this to run before it can run

### Running the actual telegram bot
```
python bot.py
```