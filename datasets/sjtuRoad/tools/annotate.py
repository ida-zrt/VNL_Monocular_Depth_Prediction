import os
import glob
import random

dataDirs = glob.glob('../data*')
annotateDir = '../annotations/'

valSplit = 0.1

rgbPaths = []

for dataDir in dataDirs:
    frameDir = dataDir + '/pic/'
    rgbPaths.extend(glob.glob(frameDir + '*'))

depthPaths = [x.replace('pic', 'depth') for x in rgbPaths]
depthPaths = [x.replace('jpg', 'png') for x in depthPaths]

lines = [
    '"{}" "{}"\n'.format(os.path.abspath(x), os.path.abspath(y))
    for x, y in zip(rgbPaths, depthPaths)
]

valLines = random.sample(lines, int(len(lines) * valSplit))
trainLines = [x for x in lines if x not in valLines]

with open(os.path.join(annotateDir, 'train.txt'), 'w') as f:
    f.writelines(trainLines)

with open(os.path.join(annotateDir, 'val.txt'), 'w') as f:
    f.writelines(valLines)
