# -*- coding: utf-8 -*-
import pickle
import re
import pandas

generated_data = []
data = []
# negative samples
picklePath = "data/short/datasets/reuters_headlines.pickle"
with open(picklePath, "rb") as pickleFile:
    pickleData = pickle.load(pickleFile)
    print(pickleData[0])
    for line in pickleData:
        generated_data.append(["0", line])
        

# positive samples
picklePath = "data/short/datasets/humorous_oneliners.pickle"
with open(picklePath, "rb") as pickleFile:
    pickleData = pickle.load(pickleFile)
    print(pickleData[0])
    for line in pickleData:
        generated_data.append(["1", line])


for row in generated_data:
    txt = ""
    for s in row[1:]:
        txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
    data.append ((int(row[0]), txt))
pd = pandas.DataFrame(data)
pd.to_hdf('data/generated_data/dataset_binary.hdf5', 'dataset_1', mode='w', complevel=9, complib='bzip2')