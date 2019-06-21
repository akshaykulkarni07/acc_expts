import os
import csv
import numpy as np

# constant length of each example
reqd_len = 150

path = '../data/acc_only.csv'

with open(path) as f :
    reader = csv.reader(f)
    # empty list that will hold readings from one example
    readings = list()
    annotation = ''
    for row in reader :
        # if non-empty row, then append the row to readings
        if row[0] != '' :
            # add the accelerometer data only (not the annotation) since
            # it will be easier to handle during finding the average of the
            # readings for the padding values
            readings.append(row[1 : ])
            # print(row[1 : ])
            # keep the label saved for using during padding
            annotation = row[0]

        # if empty row i.e. separation between 2 examples
        # then current example has ended
        if row[0] == '' :
            # if less than 140 or more than 150 samples, then
            # discard the reading
            if len(readings) < 140 or len(readings) > 150 :
                print('readings are not between 140 and 150 : ', len(readings))
                readings = list()
                continue
            # Calculating amount of padding required
            length = len(readings)
            k = 0
            # finding equal padding from both sides
            pad_length = (reqd_len - length) // 2
            # in case unsymmetrical padding is required
            if ((pad_length * 2) + length) < reqd_len :
                k = reqd_len - ((pad_length * 2) + length)
                # print(k)

            print(length + (pad_length * 2) + k)

            # converting each accelerometer reading into float from str
            readings_ = np.array(readings, dtype = float)

            # Determining the actual padding values
            # Taking the average of the readings of that particular example
            padding_ = (np.mean(readings_, axis = 0)).tolist()
            # adding the annotation to the padding row (for consistent data)
            padding_.append(annotation)
            # print(len(padding_))

            # writing the padding and data to csv
            with open('../data/padded_data.csv', 'a') as wf :
                wrf = csv.writer(wf)
                for i in range(pad_length) :
                    wrf.writerow(padding_)
                for reading in readings :
                    reading_ = reading
                    # Adding the annotation to each row of the reading
                    # since we had initially removed it
                    reading_.append(annotation)
                    wrf.writerow(reading_)
                for i in range(pad_length + k) :
                    wrf.writerow(padding_)

            # make readings empty for next time
            readings = list()
