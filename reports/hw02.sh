sbt "run-main nlp.cb.MalletRunner --train-proportion 0.8 --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/ --folds 10 --model crf" >> hw02.local.out

sbt "run-main nlp.cb.MalletRunner --train-proportion 0.8 --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/ --folds 10 --model hmm" >> hw02.local.out

remote: sbt "run-main nlp.cb.MalletRunner --train-dir /u/chbrown/penn-tagged/pos/atis --train-proportion 0.8 --folds 10 --model hmm"
./sbt "run-main nlp.cb.MalletRunner --train-dir /u/chbrown/penn-tagged/pos/wsj/00 --test-dir /u/chbrown/penn-tagged/pos/wsj/01 --folds 1 --model hmm" >> ~/hw02.out


run-main nlp.cb.MalletRunner --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/wsj/mini00 --test-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/wsj/mini01 --folds 10 --model hmm
run-main nlp.cb.MalletRunner --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis --train-proportion 0.8 --folds 10 --model hmm
run-main MalletRunner --model-file Atis3.model --training-proportion 0.8 --mallet-file /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet


run-main cc.mallet.fst.SimpleTagger --train true --model-file Atis3.model --training-proportion 0.8 --test lab /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet
run-main cc.mallet.fst.HMMSimpleTagger --train true --model-file Atis3.model --training-proportion 0.8 --test lab /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet

/Users/chbrown/Dropbox/ut/nlp/cs338-code/mallet-2.0.6/src/cc/mallet/fst
/Users/chbrown/Dropbox/ut/nlp/homework/src/main/java/mallet-2.0.7/src/cc/mallet/fst
