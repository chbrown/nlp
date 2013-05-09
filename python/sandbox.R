d = read.csv('/Users/chbrown/Downloads/05/logistic/ex2data1.txt',
                header=F, col.names=c('X', 'Y', 'Z'))

plot(d$X, d$Y, pch=d$Z+2, log='xy')
d$XY = d$X * d$Y

glm(d$Z ~ d$X + d$Y + d$XY, family=binomial(link = "probit"))

logit
