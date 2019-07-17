'''
Script to convert previously annotated data to a form that
can be annotated according to the new rules

It removes the overlap in the readings and the empty rows

I was doing this manually earlier and realized that writing a script
is faster than manually doing this for a single game data csv file.

Future improvement : Code to enter the path to original file
in the arguments while running the script, instead of manually
entering in the code. TODO : LOW-PRIORITY
'''

import os
import numpy as np
import csv
import sys

path = '/home/akshay/Desktop/acc-experiments/data/annotated_csv/_2016-11-26-18-36-15_expb_Player.csv'
out_path = '/home/akshay/Projects/acc_expts/data/new_annotated_data/_2016-11-26-18-36-15_expb_Player.csv'

existence = os.path.isfile(out_path)

# Exit the program if the output file already exists, since then
# risk appending to any already annotated file
if existence :
    print('Output file already exists. Please delete the file before running again')
    sys.exit()

with open(path) as f :
    reader = csv.reader(f)

    # empty list to hold all the readings
    readings = list()
    # initializing time to -1 so that even zero timestamped reading
    # can be taken in
    time = float(-1)
    flag = True
    # iterating over all rows of the csv
    for row in reader :
        # current timestamp
        try :
            t = float(row[0])
        # the first and empty rows have labels which throws a ValueError
        # since they are strings that cannot be converted to float
        # so, we add this to bypass that
        except ValueError :
            # this flag part is used to get the first row in the final csv
            # and also ensures that other empty rows don't get in
            if flag == True :
                readings.append(row)
                flag = False
            continue

        # if current timestamp is larger than previous,
        # then we need to append the reading to the list
        if t > time :
            readings.append(row)
            # update previous time to be current time
            time = float(row[0])

        # otherwise, we discard that reading (no else required)
        # and also don't update the previous time
        # else :

    # saving the readings to csv
    with open(out_path, 'a') as wf :
        wrf = csv.writer(wf)
        for reading in readings :
            wrf.writerow(reading)
