import os
import sys
import csv

# constant length of each example
reqd_len = 50

path = '../data/new_data.csv'

with open(path) as f :
    reader = csv.reader(f)
    # empty list to hold readings from one example
    readings = list()

    time = 0
    label = ''
    for row in reader :
        # if reading is to be continued i.e. both timestamp increases and label stays the same
        if row[0] >= time and row[1] == label :
            readings.append(row[2 : ])
            annotation = row[1]
            time = row[0]

        # if timestamp value reduces, it means start of new example
        # also, if label changes, it means start of new example
        else :
            # we need to integer divide the number of readings by `reqd_len`
            # and then segment the data into those many examples
            num_data = len(readings) // reqd_len
            # if less than required length, discard the readings
            if num_data == 0 :
                print('readings are less than ', reqd_len)
                # prepare for taking next block of data
                readings = list()
                label = row[1]
                time = row[0]
                continue

            # calculating the amount of padding required
            length = len(readings)
            k = 0
            pad_length = (reqd_len * num_data) - length
            # if too much padding is required, discard the excess readings
            if pad_length > (num_data * 10) :
                # TODO

            label = row[1]
            time = row[0]
