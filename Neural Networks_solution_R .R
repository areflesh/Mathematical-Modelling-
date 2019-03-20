library(dplyr)
library(neuralnet)
library(imager)
library(h2o)
img<-load.image("Fig_hidden_Nodes_5_level_9_ID_8.png") %>% grayscale
plot(img)
imgdata<-as.data.frame(img)
#Splitting dataset for training and prediction subsets 
traindata<-filter(imgdata, x<226|x>276)
#For prediction we take hidden part
testdata<-filter(imgdata,x>226&x<276)
#Initiaization of cluster
h2o.init(nthreads=-1,max_mem_size='1G')
trainHex<-as.h2o(traindata)
testHex<-as.h2o(testdata)
#Calculating of model
rfHex <- h2o.deeplearning(y="value", hidden = c(600,600), 
                                epochs = 60,
                               training_frame=trainHex)
#Predicting of color of hidden part based on X and Y coordinates 
resultdata<-as.data.frame(h2o.predict(rfHex,testHex))
result<-data.frame(x=testdata$x, y=testdata$y, value=round(resultdata))
names(result)<-c("x","y","value")
#Final shape 
imgresult<-rbind(traindata, result)
plot(as.cimg(imgresult))
h2o.shutdown(prompt=FALSE)
