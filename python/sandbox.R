library(ggplot2)
library(grid)
# d = read.csv('/Users/chbrown/Downloads/05/logistic/ex2data1.txt',
#                 header=F, col.names=c('X', 'Y', 'Z'))
# plot(d$X, d$Y, pch=d$Z+2, log='xy')
# d$XY = d$X * d$Y
# glm(d$Z ~ d$X + d$Y + d$XY, family=binomial(link = "probit"))

reportpath = '/Users/chbrown/Dropbox/ut/nlp/homework/reports'
setwd(reportpath)

baseline = read.table('../python/det/baseline.tsv', header=T)
ggplot(baseline, aes(x=total_articles_count, y=correct / total)) + 
  geom_line()
  

svm = read.table('../python/det/svm.tsv', header=T)
ggplot(svm[svm$feature_function=='full',], aes(
       x=total_articles_count,
       y=correct / total,
       colour=kernel)) + 
  geom_line() +
  labs(title="SVM - Full feature set", colour='Kernel',
       x='Total articles used (train + test)', y='Accuracy') +
  
  annotate("text", 1500, .695, label="Baseline") +
  geom_hline(yintercept=.691) +
  theme_grey(base_size=18) +
  theme(
    plot.title=element_text(vjust=1),
    axis.title.x=element_text(vjust=-.2),
    axis.title.y=element_text(vjust=.33))
  
ggplot(svm[svm$kernel=='polynomial',], aes(
  x=total_articles_count,
  y=correct / total,
  colour=feature_function)) + 
  geom_line() +
  labs(title="SVM (polynomial) - Features", colour='Feature function',
       x='Total articles used (train + test)', y='Accuracy') +
  
  annotate("text", 1500, .705, label="Baseline") +
  geom_hline(yintercept=.691) +
  theme_grey(base_size=18) +
  theme(
    plot.title=element_text(vjust=1),
    axis.title.x=element_text(vjust=-.2),
    axis.title.y=element_text(vjust=.33))

crf = read.table('../python/det/crf.tsv', header=T)
ggplot(crf, aes(
  x=total_articles_count,
  y=correct / total,
  colour=feature_function)) + 
  geom_line() +
  labs(title="CRF - Feature comparison", colour='Feature function',
       x='Total articles used (train + test)', y='Accuracy') +
  
  annotate("text", 1500, .705, label="Baseline") +
  geom_hline(yintercept=.691) +
  theme_grey(base_size=18) +
  theme(
    plot.title=element_text(vjust=1),
    axis.title.x=element_text(vjust=-.2),
    axis.title.y=element_text(vjust=.33))
