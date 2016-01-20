makePieChartFromFanovaOutput<-function(textToProcessPath){
  file1 = file(description = textToProcessPath, open='r' )
  
  lines = readLines(file1)
  library(stringr)
  
  toPlot=data.frame(0,0)
  for(line in lines){
    if(grepl('\\:\\s(\\w+)$', line)){
      mi = str_extract(pattern = '\\:\\s(\\w+)$', string = line)
      mi=gsub(pattern = ': ', replacement = '',x = mi)
      mj = str_extract(pattern = '^\\d+\\.\\d+', string = line)
      mj = as.numeric(mj)
      print(c(mi,mj))
      toPlot = rbind(toPlot, c(mi,mj))
    }
  }
  toPlot=toPlot[-1,]
  names(toPlot) = c('component','percent')
  toPlot$percent<-as.numeric(toPlot$percent)
  toPlot
  higherOrder<-100-sum(toPlot$percent)
  higherOrder
  toPlot<-rbind(toPlot, c('HigherOrder', higherOrder))
  toPlot$percent<-as.numeric(toPlot$percent)
  
  substrRight <- function(x, n){
    substr(x, nchar(x)-n+1, nchar(x))
  }
  
  shortfilename = substrRight(textToProcessPath, 50)
  toPlot
  pie(x=toPlot$percent, labels=toPlot$component, main=shortfilename)
}

textToProcessPath = "/Users/shanework/Desktop/hulk/home/shane/deepHAR/eval/analysisScript/18janfANOVAresults/18janOPPCNN4/variableImportances.txt"

makePieChartFromFanovaOutput(textToProcessPath)

makePieChartFromFanovaOutput()