import os
import struct
import data_dir


def get_file_name(file_dir, file_type):
    """
    :遍历指定目录下的所有指定类型的数据文件
    :file_dir: 此目录下包含.eeg原始数据文件，.vhdr文件(含mark)和.vmrk文件
    :file_type: 指定需要找到的文件类型

    :返回
    :file_names: 指定文件的绝对路径
    """

    file_names = []

    for root, dirs, files in os.walk(file_dir, topdown=False):
        for file in files:
            if file_type in file:
                file_names.append(os.path.join(root, file))

    return file_names


def read_eeg_file(file_dir, mark, point_boundary, name_index, trail_indices):
    for trail_index in range(trail_indices):
        if mark[trail_index + 1] != 0:
            with open(file_dir, 'rb') as f:
                raw = f.read()
                raw_eeg = struct.unpack('{}f'.format(int(len(raw) / 4)), raw)
                data = raw_eeg[point_boundary[trail_index] * 35: (point_boundary[trail_index + 1] + 1) * 35]
                data = struct.pack('{}f'.format(int(len(data))), *data)

            current_name_index = name_index[trail_index]
            f = open(data_dir.segmentation_dir + '\\raw_data' + '\\' + current_name_index + '.eeg', 'wb')
            f.write(data)
            f.close()



def read_mark_txt(file_dir):
    mark = []
    point_boundary = []
    with open(file_dir, 'r') as f:
        for line in f:
            if line[0:2] == 'Mk':
                mark.append(int(line[3:].split(',')[1]))
                point_boundary.append(int(line[3:].split(',')[2]))

    mark = mark[1:]
    point_boundary = point_boundary[1:]
    trail_indices = int(len(point_boundary) - 1)

    row = 0  # 用来记录读到第几行了
    txt = ''  # 先创建一个空字符串，用来存待会改写后的文本
    name_index = []  # 用于记录当前正在处理被试者的第几段子试验
    datas_length = []  # 用于记录每个子试验采集数据个数即points

    for trail_index in range(trail_indices):
        with open(file_dir, 'r') as f:
            for line in f:
                if row <= 5 and row != 3:
                    txt = txt + line
                if row == 3:
                    current_name_index = line[9:-5] + '{0:0>3}'.format(trail_index+1)  #  控制编号以001， 002，。。。， 025，。。。 113形式
                    txt = txt + line[:9] + current_name_index + '.eeg' + '\n'
                row += 1
            row = 0
            trail_point = [point_boundary[trail_index], point_boundary[trail_index+1]]
            for mk in range(len(trail_point)):
                txt = txt + 'MK' + str(mk+1) + '=Stimulus,' + str(mk+1) + ',' + str(trail_point[mk]-trail_point[0]+1) + ',0,' + '\n'

        name_index.append(current_name_index)
        datas_length.append(trail_point[mk] - trail_point[0] + 1)
        if mark[trail_index+1] != 0:
            f = open(data_dir.segmentation_dir + '\\raw_data' + '\\' + current_name_index + '.vmrk', 'w')
            f.write(txt)
            f.close()
        txt = ''

    return mark, point_boundary, name_index, trail_indices, datas_length # point_boundary用于原始.eeg文件分段，name_index, trail_indices和datas_length用于改写.vhdr文件


def read_vhdr_file(file_dir, mark, name_index, trail_indices, datas_length):
    row = 0  # 用来记录读到第几行了
    txt = ''  # 先创建一个空字符串，用来存待会改写后的文本
    for trail_index in range(trail_indices):
        current_name_index = name_index[trail_index]
        if mark[trail_index+1] != 0:
            with open(file_dir, 'r') as f:
                for line in f:
                    if row == 4:
                        txt = txt + line[:9] + current_name_index + '.eeg' + '\n'
                    elif row == 5:
                        txt = txt + line[:11] + current_name_index + '.vmrk' + '\n'
                    elif row == 10:
                        txt = txt + line[:11] + str(datas_length[trail_index]) + '\n'
                    else:
                        txt = txt + line
                    row += 1
                row = 0

            f = open(data_dir.segmentation_dir + '\\raw_data' + '\\' + current_name_index + '.vhdr', 'w')
            f.write(txt)
            f.close()
        txt = ''



if __name__ == '__main__':

    mark_file_name = get_file_name(data_dir.raw_eeg_dir, '.vmrk')
    vhdr_file_name = get_file_name(data_dir.raw_eeg_dir, '.vhdr')
    eeg_file_name = get_file_name(data_dir.raw_eeg_dir, '.eeg')

    for i in range(len(eeg_file_name)):

        mark, point_boundary, name_index, trail_indices, datas_length = read_mark_txt(mark_file_name[i])
        read_vhdr_file(vhdr_file_name[i], mark, name_index, trail_indices, datas_length)
        read_eeg_file(eeg_file_name[i], mark, point_boundary, name_index, trail_indices)
