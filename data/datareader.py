import numpy as np
import csv
import sys
import os
import h5py
import simplejson as json

class data_reader:
    def __init__(self, dataset, datapath):
        if dataset == 'opportunity':
            self.data, self.id2label = self._read_opportunity(datapath)
            self.save_data()
        else:
            print('Not supported')
            sys.exit(0)

    def save_data(self):
        f = h5py.File('opportunity.h5')
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        with open('opportunity.h5.classes.json', 'w') as f:
            f.write(json.dumps(self.id2label))

    @property
    def training(self):
        return self.data['training']

    @property
    def test(self):
        return self.data['test']

    @property
    def validation(self):
        return self.data['validation']

    def _read_opportunity(self, datapath):
        files = {
            'training': [
                'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat',                               'S2-ADL5.dat', 'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat',                               'S3-ADL5.dat', 'S3-Drill.dat', 
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
            ],
            'validation': [
                'S1-ADL2.dat'
            ],
            'test': [
                'S2-ADL3.dat', 'S2-ADL4.dat',
                'S3-ADL3.dat', 'S3-ADL4.dat'
            ]
        }

        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        label2id = {str(x[0]): i for i, x in enumerate(label_map)}
        id2label = [x[1] for x in label_map]

        cols = [
            38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
        cols = [x-1 for x in cols] # labels for 18 activities (including other)

        data = {dataset: self._read_opp_files(datapath, files[dataset], cols, label2id)
                for dataset in ('training', 'validation', 'test')}

        return data, id2label

    def _read_opp_files(self, datapath, filelist, cols, label2id):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            nancnt = 0
            print('reading file %d of %d' % (i+1, len(filelist)))
            with open(datapath.rstrip('/') + '/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    # we can skip lines that contain NaNs, as they occur in blocks at the start
                    # and end of the recordings.
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(label2id[elem[-1]])
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}

if __name__ == "__main__":
    print('Reading %s from %s' % (sys.argv[1], sys.argv[2]))
    dr = data_reader(sys.argv[1], sys.argv[2])
