library(ggplot2)

tab = read.csv('hw03-results.csv')

ggplot(tab, aes(x=iteration, y=f1, colour=selection, linetype=selection)) +
  geom_path() +
  ylab("PCFG F1 Score")
