import numpy as np
import matplotlib.pyplot as plt
from util.tictoc import tic,toc

from bluehat import rocu

N = int(9001)
t = np.random.randn(N) + 1
b = np.random.randn(N)
#b[:N]=t[:N]

## Saturate:
t[t<-0.5]=-0.5
b[b<-0.5]=-0.5

## Degeneracies near the median
#t[-0.9 < t < 1.1] = 1
#b[-0.9 < t < 1.1] = 1

if N<=10:
    print("t:",np.sort(t))
    print("b:",np.sort(b))

tic("far_pdhalf...")
fapd5 = rocu.fa_pdhalf(t,b)
toc("fapd5 time: {:.3g} sec")
print("FAR@PD5=",fapd5)

tic("pd_fax...")
pdfax = rocu.pd_fax(t,b)
toc("pd_fax time: {:.3g} sec")
print("DR@FAR01=",pdfax)
      
def testroc(fcn,label,**xtra):
    tic(label+"...")
    fa,pd = fcn(t,b,**xtra)
    toc("roc time: {:.3g} sec")
    print(label+" len(fa):",len(fa))

    tic(label+" FARPDx ...")
    farpdx = rocu.fapdx(fa,pd)
    toc("x time: {:.3g} sec")
    print("FAR@PDx=",farpdx)
    
    auc = rocu.auc_fapd(fa,pd)
    print("AUC = ",auc)

    if N<=10:
        print("fa:",fa)
        print("pd:",pd)

    if N<10000:
        plt.plot(fa,pd,label=label)

    return auc

def roc_fcnfa(tgt,bkg):
    fa,pd = rocu.roc(tgt,bkg)
    return rocu.fapd_fcn_of_fa(fa,pd)

def roc_fcnpd(tgt,bkg):
    fa,pd = rocu.roc(tgt,bkg)
    return rocu.fapd_fcn_of_pd(fa,pd)

testroc(rocu.roc,"roc",conservative=False)
#testroc(roc_fcnfa,"fcnfa")
#testroc(roc_fcnpd,"fcnpd")

if 0:
    testroc(rocu.roc,"roc/C",conservative=True)
    testroc(rocu.roc_OK,"rocOK")
    auc = testroc(rocu.roc_FAIR,"rocFAIR")
    auc -= 1/(2*len(b))
    print("AUC = ",auc,"corrected")


if N<10000:
    plt.legend()
    plt.show()
