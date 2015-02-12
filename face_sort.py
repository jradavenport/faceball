# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 21:57:05 2014

@author: jradavenport
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import readline
#from select import select
import os


flist = np.loadtxt('face.lis',dtype='string')

elo_file = 'elo_score.txt'


ntrials = 100
print 'doing 100 faces'

ranks = np.zeros((3, ntrials))

if os.path.isfile(elo_file):
    elo_score = np.loadtxt(elo_file)
else:
    elo_score = np.ones(len(flist)) * 1600.0

k_lvl = 30.

for n in range(0,ntrials):
    #pick random images
    n1 = int(np.floor(np.random.rand(1)*len(flist)))
    n2 = int(np.floor(np.random.rand(1)*len(flist)))
    if n1==n2:
        if n1>1:
            n1=n1-1
        else:
            n1=n1+1

    img1 = mpimg.imread(flist[n1])
    img2 = mpimg.imread(flist[n2])
    
    # i dont know how to more gracefully show both images
    img3 = np.zeros((np.shape(img1)[0], np.shape(img1)[1]*2, 4))
    img3[:,0:np.shape(img1)[1],:] = img1
    img3[:,np.shape(img2)[1]:,:] = img2
    
    
    plt.figure()
    plt.imshow(img3)
    plt.text(100,175,'A',color='r',size=18)
    plt.text(300,175,'B',color='r',size=18)
    plt.show(block=False)
    
    x = raw_input('A or B? ')
    
    choice = '-1'
    ranks[0,n] = n1 # save the outputs
    ranks[1,n] = n2
    if x.lower() == 'a':
        choice = str(n1)
    if x.lower() == 'b':
        choice = str(n2)

    # helpful code stolen from: 
    # http://stackoverflow.com/questions/164831/how-to-rank-a-million-images-with-a-crowdsourced-sort
    WinProb1 = 1/(10**(( elo_score[n2] - elo_score[n1])/400) + 1)
    WinProb2 = 1/(10**(( elo_score[n1] - elo_score[n2])/400) + 1)
    
    if x.lower()=='a':
        elo_score[n1] = elo_score[n1] + (k_lvl * (1. - WinProb1))
        elo_score[n2] = elo_score[n2] + (k_lvl * (0. - WinProb2))
    if x.lower()=='b':
        elo_score[n2] = elo_score[n2] + (k_lvl * (1. - WinProb2))
        elo_score[n1] = elo_score[n1] + (k_lvl * (0. - WinProb1))
    if (x.lower()!='a') and (x.lower()!='b'):
        elo_score[n1] = elo_score[n1] + (k_lvl * (.5 - WinProb1))
        elo_score[n2] = elo_score[n2] + (k_lvl * (.5 - WinProb2))
    
    plt.close()

    np.savetxt(elo_file, elo_score)
    #np.savetxt('out.txt', (ranks[:,0:n]).transpose())


    outfile = open('out.txt','a')
    outfile.write( str(n1)+'  '+str(n2)+'  '+choice+' \n')    
    outfile.close()    
    
    