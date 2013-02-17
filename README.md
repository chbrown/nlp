# Homework 1

In the root directory, start up Simple Build Tool:

    sbt

Then to run on the WSJ corpus, for example, you'll need to get the WSJ corpus, and then run (within the SBT console):

    run-main nlp.cb.NgramEvaluator ../data/penn-treebank3/tagged/pos/brown/ 0.1
