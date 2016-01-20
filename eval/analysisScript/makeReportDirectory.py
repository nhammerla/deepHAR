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
    if 'meanf1' in headers:
            headers.remove('meanf1')
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
print(vis._fanova.get_parameter_names())
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

params.sort()
print(params)
#For Continuous and Integer Parameters:
for (paramX, param_desc) in params:
        print('creating %s'%param_desc+','+paramX)
        plt2 = vis.plot_marginal(paramX)
        plt2.xlabel(param_desc)
        plt2.savefig('./'+resultsFolderName+'/'+param_desc+'.jpg')
        plt2.clf()
'''

import matplotlib.pyplot as plt
def plot_pairwise_marginal(vis, param_1, param_2, lower_bound_1=0, upper_bound_1=1, lower_bound_2=0, upper_bound_2=1, resolution=20):
        dim1, param_name_1 = vis._check_param(param_1)
        dim2, param_name_2 = vis._check_param(param_2)

        grid_1 = np.linspace(lower_bound_1, upper_bound_1, resolution)
        grid_2 = np.linspace(lower_bound_2, upper_bound_2, resolution)
        
        zz = np.zeros([resolution * resolution])
        for i, y_value in enumerate(grid_2):
            for j, x_value in enumerate(grid_1):
                zz[i * resolution + j] = vis._fanova._get_marginal_for_value_pair(dim1, dim2, x_value, y_value)[0]

        zz = np.reshape(zz, [resolution, resolution])

        display_grid_1 = [vis._fanova.unormalize_value(param_name_1, value) for value in grid_1]
        display_grid_2 = [vis._fanova.unormalize_value(param_name_2, value) for value in grid_2]

        display_xx, display_yy = np.meshgrid(display_grid_1, display_grid_2)


        plttype='contour'
        if plttype=='contour':
            X = display_xx
            Y = display_yy
            Z = zz

            plt.figure()
            CS = plt.contour(X, Y, Z)
            return plt
        if plttype=='multi3d':
           from mpl_toolkits.mplot3d import axes3d
           from matplotlib import cm

           X = display_xx
           Y = display_yy
           Z = zz
           ax = fig.gca(projection='3d')
           ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
           cset = ax.contourf(X, Y, Z, zdir='z')#, offset=-100, cmap=cm.coolwarm)
           cset = ax.contourf(X, Y, Z, zdir='x')#, offset=-40, cmap=cm.coolwarm)
           cset = ax.contourf(X, Y, Z, zdir='y')#, offset=40, cmap=cm.coolwarm)
           return plt
        if plttype=='contour3d':
            fig = plt.figure()
            ax = Axes3D(fig)
            surface = ax.contour(display_xx, display_yy, zz, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
            ax.set_xlabel(param_name_1)
            ax.set_ylabel(param_name_2)
            ax.set_zlabel("Performance")
            fig.colorbar(surface, shrink=0.5, aspect=5)
        #return {'xx':display_xx, 'yy':display_yy, 'zz':zz} 
        #return plt.contourf(display_xx, display_yy,zz)
        #return ax.plot_wireframe(display_xx, display_yy, zz).draw()
            return plt

plot3 = plot_pairwise_marginal(vis, 'X0','X1')
#show(plot3)
#plot3.xlabel('label here')
plot3.savefig('deleteme13jan1440.jpg')
#plot_pairwise_marginal(vis, 'X0','X1')


#vis.create_most_important_pairwise_marginal_plots(resultsFolderName, 5)
'''
os.system('killall java')
