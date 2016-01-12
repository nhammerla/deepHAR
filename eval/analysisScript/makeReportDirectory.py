#Usage: python makeReportDirectory.py CSVFILENAME RESULTSFOLDERNAME MINF1SCORE QUANTILETOCOMPARE
#For now, the CSV file must have a header row, and be in the same directory as this file.
hasHeader = True
import csv

def headerList(csvFileName):
    #My own way of extracting headers:
    file1 = open(csvFileName, 'r') 
    reader1 = csv.reader(file1)
    headers=reader1.next()
    if headers[0]=='id':
            headers=headers[2:]
    if 'mean f1' in headers:
            headers.remove('mean f1')
    headers = [('X'+str(i), headers[i]) for i in range(0,len(headers))]
    print(headers)
    file1.close()
    return headers

import sys

argsList = sys.argv
csvFileName = argsList[1] 
resultsFolderName = argsList[2]
minmeanF1score = argsList[3]
quantileToCompare = argsList[4]

import os
if hasHeader:
    os.system("Rscript process.R "+csvFileName+" "+minmeanF1score)

os.system('mkdir '+resultsFolderName+'  > /dev/null 2>&1')
headers = headerList(csvFileName)
csvFileName = 'noHeader'+csvFileName
print(csvFileName)
os.system('python fanovaPrintOutput.py '+csvFileName+' > ./'+resultsFolderName+'/variableImportances.txt'+' '+quantileToCompare)

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from pyfanova.fanova_from_csv import FanovaFromCSV
import inspect
f = FanovaFromCSV(csvFileName)#, improvement_over="QUANTILE", quantile_to_compare=0.25)

from pyfanova.visualizer import Visualizer

vis = Visualizer(f)
#vis.create_all_plots(resultsFolderName)


params = headers
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pyfanova.fanova_from_csv import FanovaFromCSV

f = FanovaFromCSV("noHeaderall_dnn.csv")
from pyfanova.visualizer import Visualizer
vis = Visualizer(f)

#For Continuous and Integer Parameters:
for (paramX, param_desc) in params:
        print('creating %s'%param_desc)
        plt2 = vis.plot_marginal(paramX)
        plt2.xlabel(param_desc)
        plt2.savefig('./'+resultsFolderName+'/'+param_desc+'.jpg')




vis.create_most_important_pairwise_marginal_plots(resultsFolderName, 5)

os.system('killall java')
