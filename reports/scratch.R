library(ggplot2)

tab = read.csv('hw03-results.csv')

ggplot(tab, aes(x=iteration, y=f1, colour=selection, linetype=selection)) +
  geom_path() +
  ylab("PCFG F1 Score")

+ scale_fill_continuous(limits=c(-1, 1), breaks=seq(-1,1,by=0.25))
