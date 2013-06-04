#options(stringsAsFactors = TRUE)
admis = '/Users/chbrown/corpora/admissions-selections/definites.tsv'
admissions = read.delim(admis, header=TRUE, stringsAsFactors=F)
admissions$gpa = as.numeric(admissions$gpa)
admissions = admissions[!is.na(admissions$sat) &
                        !is.na(admissions$gpa) & 
                        !admissions$gpa == 0 & 
                        !admissions$wc == 0,]
admissions$dets = admissions$indefinites + admissions$definites
admissions$det_ratio = admissions$dets/admissions$wc
with(admissions, smoothScatter(gpa, dets/wc))
?smoothScatter

#View(admissions)
#admissions$gpa[192] == ''

#head(admissions)
intercept.lm = lm(gpa ~ 1, data=admissions)
summary(intercept.lm)

wc.lm = lm(gpa ~ wc, data=admissions)
summary(wc.lm)

gpa.lm = lm(gpa ~ I(indefinites/wc) + I(definites/wc) + wc, data=admissions)
summary(gpa.lm)
anova(gpa.lm)

gpa.aov = aov(gpa ~ I(indefinites/wc) + I(definites/wc) + wc + sat,
              data=admissions)
summary(gpa.aov)
plot(gpa.aov)

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


