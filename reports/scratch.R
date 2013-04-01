library(ggplot2)

setwd('/Users/chbrown/Dropbox/ut/nlp/homework/reports')

tab = read.csv('hw03-results.csv')

ggplot(tab, aes(x=iteration, y=f1, colour=selection, linetype=selection)) +
  geom_path() +
  ylab("PCFG F1 Score")


origparse2 = read.csv('origparse2.csv')

origparse2.rev = read.csv('origparse2-rev.csv')
origparse2.rev$selection = paste(origparse2.rev$selection, '-rev')
origparse2 = rbind(origparse2, origparse2.rev)

ggplot(origparse2, aes(x=iteration, y=f1, colour=selection, linetype=selection)) + geom_path() +
    ylab("PCFG F1 Score") + xlab("Iteration")

ggplot(origparse2, aes(x=total, y=f1, colour=selection, linetype=selection)) + geom_path() +
  ylab("PCFG F1 Score") + xlab("Total training words used")
