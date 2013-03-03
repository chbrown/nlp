target/start nlp.cb.MalletRunner --train-proportion 0.8 --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/ --folds 10 --model crf

target/start nlp.cb.MalletRunner --train-proportion 0.8 --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/ --folds 10 --model hmm

target/start nlp.cb.MalletRunner --train-dir /u/chbrown/penn-tagged/pos/atis --train-proportion 0.8 --folds 10 --model hmm
target/start nlp.cb.MalletRunner --train-dir /u/chbrown/penn-tagged/pos/wsj/00 --test-dir /u/chbrown/penn-tagged/pos/wsj/01 --folds 1 --model hmm

target/start nlp.cb.MalletRunner --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/wsj/mini00 --test-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/wsj/mini01 --folds 10 --model hmm
target/start nlp.cb.MalletRunner --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis --train-proportion 0.8 --folds 10 --model hmm
target/start MalletRunner --model-file Atis3.model --training-proportion 0.8 --mallet-file /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet

target/start cc.mallet.fst.SimpleTagger --train true --model-file Atis3.model --training-proportion 0.8 --test lab /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet
target/start cc.mallet.fst.HMMSimpleTagger --train true --model-file Atis3.model --training-proportion 0.8 --test lab /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet

sbt
> add-start-script-tasks
> start-script
> [CTRL-D]

export DATA=/Users/chbrown/Dropbox/ut/nlp/data
export DATA=/u/chbrown/data

# run atis hmm
target/start nlp.cb.MalletRunner --train-proportion 0.8 --model hmm --folds 10 \
  --train-dir $DATA/penn-treebank3/tagged/pos/atis \
  >> ../hw02.out
# run atis crf
target/start nlp.cb.MalletRunner --train-proportion 0.8 --model crf --folds 10 \
  --train-dir $DATA/penn-treebank3/tagged/pos/atis \
  >> ../hw02.out
# run atis crf with extras
target/start nlp.cb.MalletRunner --train-proportion 0.8 --model crf --extras --folds 10 \
  --train-dir $DATA/penn-treebank3/tagged/pos/atis \
  >> ../hw02.out

# run wsj 00 wsj 01 hmm
target/start nlp.cb.MalletRunner --model hmm \
  --train-dir $DATA/penn-treebank3/tagged/wsj/00 \
  --test-dir $DATA/penn-treebank3/tagged/wsj/01 \
  >> ../hw02.out


