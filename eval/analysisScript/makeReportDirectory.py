#Usage: python CSVFILENAME RESULTSFOLDERNAME MINF1SCORE
#For now, the CSV file must have a header row, and be in the same directory as this file.
hasHeader = True

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
csvFileName = 'noHeader'+csvFileName
print(csvFileName)
os.system('python fanovaPrintOutput.py '+csvFileName+' > ./'+resultsFolderName+'/variableImportances.txt'+' '+quantileToCompare)

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from pyfanova.fanova_from_csv import FanovaFromCSV
f = FanovaFromCSV(csvFileName)#, improvement_over="QUANTILE", quantile_to_compare=0.25)

from pyfanova.visualizer import Visualizer

vis = Visualizer(f)
vis.create_all_plots(resultsFolderName)
vis.create_most_important_pairwise_marginal_plots(resultsFolderName, 5)

os.system('killall java')
