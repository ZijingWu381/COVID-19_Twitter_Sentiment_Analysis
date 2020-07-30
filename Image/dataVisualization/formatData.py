from csv import writer
from csv import reader
# Open the input_file in read mode and output_file in write mode
for i in range(2,3):
    with open('data/transition/data_coordinate_score_' + str(i) + '.csv', 'r') as read_obj, \
            open('data/transition/date'+ str(i) + '.csv', 'w') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        next(csv_reader)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # header
        csv_writer.writerow(["", "long", "score","lat","n"])

        # Read each row of the input csv file as list
        count = 0
        for row in csv_reader:     
            # run 10 times for testing
            count = count + 1
            if (count == 10):
                break

            long = row[1].split(",")[0]
            lat = row[1].split(",")[1]
            row[1] = long
            row.append(lat)
            row.append(10)
            csv_writer.writerow(row)
        

