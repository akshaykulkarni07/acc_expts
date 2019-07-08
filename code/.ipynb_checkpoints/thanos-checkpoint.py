# Removes half of the examples from input csv file 
# Hence named thanos (Refer Marvel's Avengers Infinity Series for more information)

import os
import csv
import numpy as np

path = '../data/acc_only.csv'

with open(path) as f : 
    reader = csv.reader(f)
    
    readings = list()
    i = 0
    for row in reader :
        # If we come across a non-empty row
        if row[0] != '' :  
            # alternate reading to be saved
            # add it to the list 
            if i % 2 == 0 : 
                readings.append(row)
                
            # otherwise don't add
            
        # if empty row, then current example has ended
        elif row[0] == '' : 
            # increment i
            i = i + 1
            
            # if readings list has some content, only then we need to write to the file
            if readings != list() : 
                # write readings to other csv file
                with open('../data/alternate_data.csv', 'a') as wf :
                    wrf = csv.writer(wf)
                    for reading in readings : 
                        wrf.writerow(reading)
                    
                    empty_row = ['', '', '', '']
                    # adding empty row
                    wrf.writerow(empty_row)

                # re-initialize for next example
                readings = list()