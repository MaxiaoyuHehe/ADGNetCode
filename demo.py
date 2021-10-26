import os
for i in range(2015):
    os.rename('wQ%04d' % i, 'wwQ%04d.mat' % i)
    os.rename('wS%04d'%i,'wwS%04d.mat'%i)
