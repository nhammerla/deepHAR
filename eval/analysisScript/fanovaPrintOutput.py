import sys

argsList = sys.argv
print(argsList[1])
print('--')
from pyfanova.fanova_from_csv import FanovaFromCSV
f = FanovaFromCSV(argsList[1], improvement_over="QUANTILE", quantile_to_compare=0.25)
f.print_all_marginals()
