import sys

argsList = sys.argv
print('--')
quantile1 = argsList[2]
from pyfanova.fanova_from_csv import FanovaFromCSV
f = FanovaFromCSV(argsList[1], improvement_over="QUANTILE", quantile_to_compare=quantile1)
f.print_all_marginals()
