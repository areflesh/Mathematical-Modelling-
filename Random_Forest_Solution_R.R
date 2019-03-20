library(dplyr)
library(neuralnet)
library(imager)
library(h2o)
img<-load.image("tht.png") %>% grayscale
imgdata<-as.data.frame(img)
#I choose dataset for training of manualy. We can think about automatization
#For training we take both parts of hidden shape 
traindata<-filter(imgdata, x<188|x>260)
#For test we take hidden part
testdata<-filter(imgdata,x>188&x<260)
#Initiaization of cluster
h2o.init(nthreads=-1,max_mem_size='1G')
trainHex<-as.h2o(traindata)
testHex<-as.h2o(testdata)
#Calculating of model
rfHex <- h2o.randomForest(y="value", 
                               ntrees = 1000,
                               max_depth = 30,
                               nbins_cats = 1115,
                               training_frame=trainHex)
#Predicting of color of hidden part based on X and Y coordinates 
resultdata<-as.data.frame(h2o.predict(rfHex,testHex))
result<-data.frame(x=testdata$x, y=testdata$y, value=resultdata)
names(result)<-c("x","y","value")
#Final shape 
imgresult<-rbind(traindata, result, filter(imgdata,x>260))
plot(as.cimg(imgresult))
