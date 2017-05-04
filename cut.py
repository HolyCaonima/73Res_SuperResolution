import os
import numpy as np
from scipy.misc import imsave, imshow, imresize, imread

data_dir = './helen/trainset'
save_dir = './saveTri'
out_size = 200

imgs = os.listdir(data_dir)
imgs = [img for img in imgs if len(img.split('.')) and ( img.split('.')[1]=='jpg' or img.split('.')[1]=='png') ]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for im in imgs:
    data = imread(data_dir+'/'+im)
    pts = im.split('.')[0]+'.pts'
    ptsfile = open(data_dir+'/'+pts)
    ptsfile.readline()
    ptsfile.readline()
    ptsfile.readline()
    points = np.zeros([68,2])
    for i in range(68):
        li = ptsfile.readline().replace("\n","")
        li = li.split(' ')
        points[i,0]=float(li[0])
        points[i,1]=float(li[1])

    ptsfile.close()
    mi = np.min(points, 0).astype('int')
    mx = np.max(points, 0).astype('int')
    data = data[mi[1]:mx[1],mi[0]:mx[0]]
    row=data.shape[0]
    col=data.shape[1]
    if row<out_size or col<out_size:
        print "Skip one"
        continue
    if row > col:
        data = data[(row-col)/2:col+(row-col)/2,:]
    else:
        data = data[:, (col-row)/2:row+(col-row)/2]
    data = imresize(data,[out_size, out_size])
    imsave(save_dir + '/' + im, data)
