#options(stringsAsFactors = TRUE)
admis = '/Users/chbrown/Dropbox/ut/nlp/project/admissions-definiteness.tsv'
admissions = read.delim(admis, header=TRUE, stringsAsFactors=F)
admissions$gpa = as.numeric(admissions$gpa)
admissions = admissions[!is.na(admissions$gpa) & admissions$wc != 0,]
#admissions[is.na(admissions$det_ratio),]
admissions$dets = admissions$indefinites + admissions$definites
admissions$det_ratio = admissions$dets/admissions$wc

View(admissions)
#admissions$gpa[192] == ''

head(admissions)
intercept.lm = lm(gpa ~ 1, data=admissions)
summary(intercept.lm)
wc.lm = lm(sat ~ wc, data=admissions)
summary(wc.lm)
gpa.lm = lm(sat ~ det_ratio, data=admissions)
summary(gpa.lm)


gpasat.lm = lm(gpa ~ sat, data=admissions)
summary(gpasat.lm)


sd(admissions$gpa)
hist(admissions$det_ratio)
hist(admissions$gpa)
sqrt(mean(gpa.lm$residuals^2))
sd(gpa.lm$residuals)



# testing simple regression
a = 1:100
anoise = rnorm(100, a, 1)
plot(a, anoise)
summary(lm(a ~ anoise))


