from math import cos, sin


class Measurement:
    def __init__(self, range: float, bearing: float) -> None:
        self.range = range
        self.bearing = bearing

    def valid(self):
        return self.range > 0 and self.range < 15

    def x(self):
        return self.range * cos(self.bearing)

    def y(self):
        return self.range * sin(self.bearing)

def load_measurements(filename: str):
    file_lines = []
    meas_data = []
    threshold = 110
    if 'point5' in filename:
        threshold = 120
    with open(filename, 'r') as f:
        file_lines = f.readlines()

    for line in range(0, len(file_lines)):
        if line >= threshold:
            continue
        # print(line)
        line_list = file_lines[line].split(',')
        meas_data_line = []
        angle = 0
        angle_inc = 0
        for val in range(len(line_list)):
            if val < 9:
                if val == 2:
                    angle = float(line_list[val])
                if val == 3:
                    angle_inc = float(line_list[val])
                continue 
            if line_list[val] == 'inf':
                line_list[val] = 20.0
            range_val = float(line_list[val])
            meas_data_line.append(Measurement(range_val, angle))
        meas_data.append(meas_data_line)
    return meas_data
