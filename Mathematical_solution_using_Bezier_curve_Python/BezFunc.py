"""

"""

#import sys
#import math
import os
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random

sqr2 = np.sqrt(2)
relpos = [[int(1.8*np.cos(a*np.pi/4)),int(1.8*np.sin(a*np.pi/4))] for a in range(8)]

if os.path.exists("BinomialTable.bin"):
    binomial_table = np.load("BinomialTable.bin")
else:
    binomial_table = np.zeros((30, 30))
    for i in range(30):
        for j in range(30):
            binomial_table[i][j] = sp.binom(i, j).copy()
    np.save("BinomialTable.bin", binomial_table)

def sqrSum(a, b, i, j):
    """
    returns the squared distance between (a, b) and (i, j)
    """
    return (a - i)**2 + (b - j)**2

def dist(a, b, i, j):
    """
    returns cartesian distance between points (a, b) and (i, j)
    """
    return np.sqrt(sqrSum(a, b, i, j))


def check_ext(im, i, j):
    """
    Simple function to check if a given pixel is an extrem of the line to
    be fitted
    """
    neighb = 0
    count = 0
    for a in range(8):
        if (im[i+relpos[a][0], j+relpos[a][1]] and (count == 0)):
            count += 1
            neighb += 1
        else:
            count = 0
    return (neighb < 2)

def bezierFunc(ctrlP, t):
    n = len(ctrlP) - 1
    point = np.zeros((1, 2))
    for i, cp in enumerate(ctrlP):
        point += sp.binom(n, i) * (1. - t)**(n - i) * t**i * cp #ctrlP[i]
#        point += binomial_table[n, i] * (1. - t)**(n - i) * t**i * cp #ctrlP[i]
    return point


def bezierPoly(ctrlP):
    """
    Returns a list of sample points belonging to the bezier polynomial obtained
    from the set of control points ctrlP
    """
    n = len(ctrlP) - 1  #degree of the polynomial
    first = True
    for t in np.linspace(0.0, 1.0, 5 * n):
        point = bezierFunc(ctrlP, t)
        if first:   # Initialize list of points in the polynomial
            bezierPointsList = np.copy(point)
            first = False
        else:
            bezierPointsList = np.append(bezierPointsList, point, axis=0)
    return bezierPointsList

def bezier2Pixel(ctrlPoly, w, h):
    npbez = np.zeros((w,h), dtype=bool)
    t0 = 0.0
    tf = 1.0
    t = t0 + (tf - t0)*.5
    prevPoint = np.copy(ctrlPoly[0])
    point = np.rint(bezierFunc(ctrlPoly, t)[0])
    while (point != ctrlPoly[-1]).any():
        if (sqrSum(prevPoint[0], prevPoint[1], point[0], point[1]) < 1):
            t0 = t
            t = t + (tf - t)*.5
        elif (sqrSum(prevPoint[0], prevPoint[1], point[0], point[1]) > 2):
            t = t0 + (t - t0)*.5
        else:
            x, y = (int(point[0]), int(point[1]))
            x = max(1, min(w - 2, x))
            y = max(1, min(h - 2, y))
            npbez[x, y] = True
            prevPoint = point
            t0 = t
            t = t0 + (tf - t0)*.5
        point = np.rint(bezierFunc(ctrlPoly, t)[0])
    x, y = (int(point[0]), int(point[1]))
    x = max(1, min(w - 2, x))
    y = max(1, min(h - 2, y))
    npbez[x, y] = True
    return npbez

def rndBezier(deg, st, end):
    """
    Creates a random Bezier curve of degree deg between points st and end.
    Used for developing purposes.
    """
    if (deg == 0):
        print("ERROR: The bezier curve degree has to be greater than 0")
        return
    totalLength = dist(st[0], st[1], end[0], end[1])
    varLength = totalLength / deg
    controlP = np.zeros((deg + 1, 2))
    controlP[0] = np.asarray(st)
    for i in range(deg - 1):
        point = controlP[i] + 1 / deg * (np.asarray(end) - np.asarray(st))
        modVar = np.random.uniform(0, 1.5*varLength)
        angVar = np.random.uniform(0, 2 * np.pi)
        point += modVar * np.asarray([np.cos(angVar) , np.sin(angVar)])
        controlP[i + 1] = point
    controlP[-1] = np.asarray(end)
    return controlP


def initPoly(deg, st, end):
    """
    Initializes a control polynomial uniformly distributed between st and
    end points in a straight line
    """
    if (deg == 0):
        print("ERROR: The bezier curve degree has to be greater than 0")
        return
    controlP = np.zeros((deg + 1, 2))
    controlP[0] = np.asarray(st)
    for i in range(deg - 1):
        point = controlP[i] + 1 / deg * (np.asarray(end) - np.asarray(st))
        controlP[i + 1] = point
    controlP[-1] = np.asarray(end)
    return controlP

"""
WRONG IMPLEMENTATION: TO BE CORRECTED
"""

##-AEM-INI - 20/01/2019
#def costFunction(bezierList, pixelList):
#    bN = len(bezierList)
    # pN = len(pixelList)
    # totErr = 0.0
    # for bPoint in range(1, bN - 1):
    #     a, b = (bezierList[bPoint][0], bezierList[bPoint][1])
    #     i, j = (pixelList[0][0], pixelList[0][1])
    #     minErr = sqrSum(a, b, i, j)
    #     for pPoint in range(1, pN):
    #         i, j = (pixelList[pPoint][0], pixelList[pPoint][1])
    #         minErr = min(minErr, sqrSum(a, b, i, j))
    #     totErr += np.sqrt(minErr)
    #
    # print (totErr)
    # return totErr
"""
Esta funcion deberia funcionar (pero no lo hace)
"""
def costFunction_old(bezierList,pixelList,controlPoints,npIm,w,h):

    npBez = bezier2Pixel(controlPoints, w, h)

    newImage = npIm
    k = 3

    for j in range(h):
        for i in range(w):
            newImage[i,j] = npBez[i,j] + npIm[i,j]

    i = 0
    k = 2
    while(i < w - 1):

        sw_inside = -1
        sw_anterior = False

        for j in range(h):
            temp = newImage[i,j]
            if newImage[i,j] == False:

                if sw_anterior == True :
                    sw_inside = sw_inside * -1


                if sw_inside == 1:
                    newImage[i,j] = True

            sw_anterior = temp

            if j == h-1 and newImage[i,j] == True:
                if newImage[i,j-3]==True and newImage[i,j-10] ==True:

                    for j in range(h):
                        newImage[i,j] = False

        i = i + k




    newImage_pixel = getPxList(newImage)
    plt.plot([x[0] for x in newImage_pixel], [x[1] for x in newImage_pixel], marker = ".", linestyle="")
    plt.show()
    totErr = np.sum(newImage)
    print(totErr)

    return totErr

"""
Esta si funciona
"""
def costFunction(pixelList, controlPoints, npIm, w, h):

    npBez = bezier2Pixel(controlPoints, w, h)

    maskRealX = np.zeros((w,h), dtype=bool)
    maskBezX = np.zeros((w,h), dtype=bool)
    
    for x in range(w):
        switchR = False
        switchB = False
        for y in range(h):
            if npIm[x, y]:
                if not(npIm[x, y - 1]):
                    switchR = not(switchR)
            elif (switchR):
                maskRealX[x, y] = True
            if npBez[x, y]:
                if not(npBez[x, y - 1]):
                    switchB = not(switchB)
            elif (switchB):
                maskBezX[x, y] = True
     
    maskRealY = np.zeros((w,h), dtype=bool)
    maskBezY = np.zeros((w,h), dtype=bool)            
    
    for y in range(h):
        switchR = False
        switchB = False
        for x in range(w):
            if x == controlPoints[0][0]:
                switchR = not(switchR)
                switchB = not(switchB)
            if npIm[x, y]:
                if not(npIm[x - 1, y]):
                    switchR = not(switchR)
            elif (switchR):
                maskRealY[x, y] = True
            if npBez[x, y]:
                if not(npBez[x - 1, y]):
                    switchB = not(switchB)
            elif (switchB):
                maskBezY[x, y] = True
    
    npInnerR = np.logical_and(maskRealX, maskRealY)
    npInnerB = np.logical_and(maskBezX, maskBezY)
    
    between = np.logical_xor(npInnerR, npInnerB) 

#    plt.imshow(between.T, cmap=plt.cm.gray)

    return np.count_nonzero(between)

##-AEM-FIN
def alphaReduce(invar, limit):
    val = invar * 0.5
    if val >= limit:
        return val
    else:
        return limit

def alphaIncrease(invar, limit):
    val = invar * 1.1
    if val >= limit:
        return limit
    else:
        return val


def imageProcess(PL, PR, imArr, controlPoly):
    bezP = bezierPoly(controlPoly)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(10, 450)
    ax.set_ylim(0, 250)
    ax.plot([x[0] for x in PL], [x[1] for x in PL],linestyle="",marker = ".")
    ax.plot([x[0] for x in PR], [x[1] for x in PR],linestyle="",marker = ".")
    ax.plot([x[0] for x in controlPoly], [x[1] for x in controlPoly], marker = "x")
    ax.plot([x[0] for x in bezP], [x[1] for x in bezP], marker = "x")
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imArr.append(image)
    return


def fitBezier(deg, pixelList, start, end, npIm ,w ,h ,alphaMax ,count_max, PL, PR, imArr):
    controlPoly = initPoly(deg, start, end)
    initCost = costFunction(pixelList, controlPoly, npIm, w, h)
    print(initCost)
    gradCost = np.zeros((deg + 1, 2))
    itNum = 0
    count = 0
    alpha = np.sqrt(initCost) * (deg - 1) / 2
    print(f'New alpha = {alpha}') 
    beta = alpha / 2
    print(f'New beta = {beta}')
#    plt.plot([x[0] for x in controlPoly], [x[1] for x in controlPoly], marker = "x")
#    plt.show()
    while (True):
        for cPoint in range(1, deg):
            for coord in range(2):
                varContPoly = np.copy(controlPoly)
                varContPoly[cPoint][coord] += alpha
                varCost = (costFunction(pixelList,varContPoly,npIm,w,h) - initCost) / alpha
                gradCost[cPoint][coord] = varCost
        gradMod = np.linalg.norm(gradCost)
        if (gradMod > 0):
            factor = beta / gradMod
            newContPoly = np.copy(controlPoly) - np.copy(gradCost) * factor
            newCost = costFunction(pixelList,newContPoly,npIm,w,h)
        else:
            alpha = alphaIncrease(alpha, np.sqrt(initCost) * (deg - 1) / 5)
            beta = alpha
            print(f'New alpha = {alpha}')
            print(f'New beta = {beta}')
            continue
        costChange = newCost - initCost
        if ( costChange >= 0 ):
            count += 1
            beta = alphaReduce(beta, 1)
            print(f'New beta = {beta}')
            alpha = alphaReduce(alpha, 1)
            print(f'New alpha = {alpha}')
            print(f'(Count = {count})')
            if (count > count_max):
                return controlPoly
        elif (itNum < 100):
            print(f'Iteration: {itNum}  {count} \tCost: {newCost}')
            if costChange < 0.1 * initCost:
                beta = alphaIncrease(beta, alpha)
            count = 0
            itNum += 1
            controlPoly = np.copy(newContPoly)
            initCost = newCost
            alpha = np.sqrt(initCost) * (deg - 1) / 5
            beta = alpha
            print(f'New alpha = {alpha}')
            print(f'New beta = {beta}')
            imageProcess(PL, PR, imArr, controlPoly)
#            plt.show()
        else:
            print("Max iterations reached without convergence")
            return newContPoly

def getPxList(npMatrix):
    pxList = []
    for i in range(len(npMatrix)):
        for j in range(len(npMatrix[0])):
            if npMatrix[i,j] ==  True:
                pxList += [[i, j]]
    return pxList

"""
Modifications of img2Pixel:
    We add a new argument (pos), if pos = "right" it extracts the pixelmap of the right part of the shape.
    The same with "left", and "none" for extracting all the pixelmap
"""
#def img2Pixel(image):
#    w, h = image.size
#    pix = image.load()
#    npimg = np.zeros((w,h), dtype=bool)
#    for i in range(w):
#        for j in range(h):
#            if pix[i, j] == 0:
#                npimg[i, j] = True
#    return (w, h, npimg)

def img2Pixel(image,pos):

    w, h = image.size
    pix = image.load()
    npimg = np.zeros((w,h), dtype=bool)
    if pos == "right":
        for i in range(w):
            for j in range(h):
                if i > 250:
                #print(pix[i,j])
                    if pix[i, j][1] != 255:
                    #print("entra")
                        npimg[i, j] = True
                else:
                    npimg[i, j] = False
        return (w, h, npimg)

    if pos == "left":
        for i in range(w):
            for j in range(h):
                if i < 250:
                #print(pix[i,j])
                    if pix[i, j][1] != 255:
                    #print("entra")
                        npimg[i, j] = True
                else:
                    npimg[i, j] = False
        return (w, h, npimg)

    if pos == "none":
        for i in range(w):
               for j in range(h):
                    if pix[i, j] == 0:
                        npimg[i, j] = True
        return (w, h, npimg)

"""
AvgPixelValue: Returns the sum of the number of black neighbour of each pixel
avgPixelMap: Returns a reduced pixel map using AvgPixelValue. If a pixel has more than 6 neighbours colored
in black, returns True. This is used to reduce the number of black pixels due to the problems with extracting the
extreme points.
"""
def AvgPixelValue(npImg,i,j):
    count = 0
    for q in [0,1,-1]:
        for k in [0,1,-1]:
            if k == 0 and q == 0:
                pass
            else:
                try:
                    if npImg[i+k,j+q] == True:
                        count = count + 1
                except IndexError:
                    pass
    return count

def avgPixelMap(npImg,w,h):

    npPixelMap = np.zeros((w,h), dtype=bool)
    for i in range(w):
        for j in range(h):
            if AvgPixelValue(npImg,i,j) > 6:
                npPixelMap[i,j] = True

    return npPixelMap

def findExtremePts(npPixelMapList,npPixelMap):
    extrems = []
    for elem in npPixelMapList:
        if check_ext(npPixelMap, elem[0], elem[1]):
                    extrems += [[elem[0], elem[1]]]

    return extrems

"""
End of Modifications
"""

def CompareCurves(deg,start,end,w,h,imPixelList,npImg,alpha,count_max, PL, PR, imArr):
    controlPoly = initPoly(deg, start, end)
#    bezList = bezierPoly(controlPoly)
    npBez = bezier2Pixel(controlPoly, w, h)
    bezPixelList = getPxList(npBez)

#    plt.plot([x[0] for x in controlPoly], [x[1] for x in controlPoly], marker = "x")
#    plt.plot([x[0] for x in imPixelList], [x[1] for x in imPixelList], linestyle="", marker = ".")
#    plt.plot([x[0] for x in bezPixelList], [x[1] for x in bezPixelList], linestyle="", marker = ".")
#    plt.show()

    controlPoly = fitBezier(deg, imPixelList, start, end, npImg, w, h, alpha, count_max, PL, PR, imArr)
    npBez = bezier2Pixel(controlPoly,w,h)
    bezPixelList = getPxList(npBez)

#    plt.plot([x[0] for x in controlPoly], [x[1] for x in controlPoly], marker = "x")
#    plt.plot([x[0] for x in imPixelList], [x[1] for x in imPixelList], linestyle="", marker = ".")
#    plt.plot([x[0] for x in bezPixelList], [x[1] for x in bezPixelList], linestyle="", marker = ".")
    print("End of fitting")
    return controlPoly

"""
Interpolation related functions
"""

def getDerivValues(contPoly, ext):
    """
    Derivative values at t = ext (0 or 1)
    """
    if not((ext == 0) or (ext == 1)):
        print("Extrem value has to be 0 or 1")
        return
    deg = len(contPoly) - 1
    derVector = np.zeros((deg, 2))
    for d in range(deg):
        preF = deg + 2
        for i in range(d+2):
            preF = preF * (deg + 1 - i)
            indexP = (1 - ext) * i - ext * (1 + i)
            exponent = i + (1 - ext) * (d + 1)
            derVector[d] += sp.binom(d+1, i)*contPoly[indexP]*(-1)**exponent
#            derVector[d] += binomial_table[d+1, i]*contPoly[indexP]*(-1)**exponent
        preF = preF / ((deg + 1)*(deg + 2))
        derVector[d] = preF * derVector[d]
    return derVector

def interpolatePoly_old(contPolyA, contPolyB, degDer):
    """
    constructs a new control polynomial from the end of contPolyA to the
    beginning of contPolyB, keeping continuity on the degDer-derivative
    """
    m = 2 * degDer + 1
    derVectorsA = getDerivValues(contPolyA, 1)
    derVectorsB = getDerivValues(contPolyB, 0)
    newContPol = np.zeros((m + 1, 2))
    newContPol[0] = contPolyA[-1]
    newContPol[-1] = contPolyB[0]
    for i in range(degDer):
        preF = m + 1
        j = - (1 + i)
        tempA = np.zeros((1, 2))[0]
        tempB = np.zeros((1, 2))[0]
        for p in range(i + 1):
            preF = preF * (m - p)
            q = - (1 + p)
            tempA += sp.binom(i + 1, p)*newContPol[p]*(-1)**(p + i + 1)
            tempB += sp.binom(i + 1, p)*newContPol[q]*(-1)**p
#            tempA += binomial_table[i+1, p]*newContPol[p]*(-1)**(p + i + 1)
#            tempB += binomial_table[i+1, p]*newContPol[q]*(-1)**p
        preF = preF / (m + 1)
        newContPol[i+1] = (derVectorsA[i] / preF - tempA)
        newContPol[j-1] = (derVectorsB[i] / preF - tempB)*(-1)**(i + 1)
    return newContPol

def interpolatePoly_old2(contPolyA, contPolyB, degP):
    m = 2 * degP + 1
    newContPol = np.zeros((m + 1, 2))
    trimPolyA = np.copy(contPolyA[-(1 + degP):])
    trimPolyB = np.copy(contPolyB[:(1 + degP)])
    derVA = getDerivValues(trimPolyA, 1)
    derVB = getDerivValues(trimPolyB, 0)
    newContPol[0] = contPolyA[-1]
    newContPol[-1] = contPolyB[0]    
    for i in range(degP):
        preF = m + 1
        j = - (1 + i)
        tempA = np.zeros((1, 2))[0]
        tempB = np.zeros((1, 2))[0]
        for p in range(i + 1):
            preF = preF * (m - p)
            q = - (1 + p)
            tempA += sp.binom(i + 1, p)*newContPol[p]*(-1)**(p + i + 1)
            tempB += sp.binom(i + 1, p)*newContPol[q]*(-1)**p
#            tempA += binomial_table[i+1, p]*newContPol[p]*(-1)**(p + i + 1)
#            tempB += binomial_table[i+1, p]*newContPol[q]*(-1)**p
        preF = preF / (m + 1)
        newContPol[i+1] = (derVA[i] / preF - tempA)
        newContPol[j-1] = (derVB[i] / preF - tempB)*(-1)**(i + 1)
    return newContPol

def interpolatePoly(contPolyA, contPolyB, degP, degM):
    trimPolyA = np.copy(contPolyA[-(1 + max(degP,1)):])
    trimPolyB = np.copy(contPolyB[:(1 + max(degP,1))])
    m = degM
    if (m < 2 * degP + 1):
        m = 2 * degP + 1
    if (m > 2 * degP + 1):
        m = max(2, 2 * degP + 1)
    if (m == 2):
        a = np.array([trimPolyA[-1] - trimPolyA[-2], - trimPolyB[0] + trimPolyB[1]])
        b = trimPolyB[0] - trimPolyA[-1]
        try:
            x = np.linalg.solve(a.T, b)
            xpos = (trimPolyA[-2] + x[0] * (trimPolyA[-1] - trimPolyA[-2]))[0]
            if ( np.any(np.less_equal(x, 0)) ):
                m = 3
                degP = 1
            elif ( (xpos <= min(trimPolyA[-1][0], trimPolyB[0][0])) or (
                    xpos >=max(trimPolyA[-1][0], trimPolyB[0][0]))):
                m = 3
                degP = 1
        except np.linalg.LinAlgError:
            m = 3
            degP = 1
    newContPol = np.zeros((m + 1, 2))
    derVA = getDerivValues(trimPolyA, 1)
    derVB = getDerivValues(trimPolyB, 0)
    newContPol[0] = trimPolyA[-1]
    newContPol[-1] = trimPolyB[0]
    if (m == 2):
        newContPol[1] = trimPolyA[-2] + x[0] * (trimPolyA[-1] - trimPolyA[-2])
    for i in range(degP):
        preF = m + 1
        j = - (1 + i)
        tempA = np.zeros((1, 2))[0]
        tempB = np.zeros((1, 2))[0]
        for p in range(i + 1):
            preF = preF * (m - p)
            q = - (1 + p)
            tempA += sp.binom(i + 1, p)*newContPol[p]*(-1)**(p + i + 1)
            tempB += sp.binom(i + 1, p)*newContPol[q]*(-1)**p
#            tempA += binomial_table[i+1, p]*newContPol[p]*(-1)**(p + i + 1)
#            tempB += binomial_table[i+1, p]*newContPol[q]*(-1)**p
        preF = preF / (m + 1)
        newContPol[i+1] = (derVA[i] / preF - tempA)
        newContPol[j-1] = (derVB[i] / preF - tempB)*(-1)**(i + 1)
    if (m < degM):
        newContPol = subDividePoly(newContPol, int(degM / m + 0.5) )
    return newContPol

def closeFigure(contPolyL, contPolyR, degDer, m):
    """
    Takes two vertical halves of the figure and interpolates 2 polynomials
    between them
    """
    clockwiseL = contPolyL[0][1] < contPolyL[-1][1]
    clockwiseR = contPolyR[0][1] > contPolyR[-1][1]
    orderedPolyR = np.zeros((len(contPolyR), 2))
    swap = not(clockwiseL == clockwiseR)
    for i in range(len(contPolyR)):
        if swap:
            orderedPolyR[i] = contPolyR[-1 - i]
        else:
            orderedPolyR[i] = contPolyR[i]
    polyInt1 = interpolatePoly(contPolyL, orderedPolyR, degDer, m)
    polyInt2 = interpolatePoly(orderedPolyR, contPolyL, degDer, m)
    return (polyInt1, polyInt2)
    
def subDividePoly(contPoly, factor):
    n = len(contPoly) - 1
    oldPoly = np.copy(contPoly)
    if (factor == 1):
        return contPoly
    for i in range((factor - 1)*n):
        nold = len(oldPoly) - 1
        m = nold + 1
        newPoly = np.zeros((m + 1, 2))
        newPoly[0] = oldPoly[0]
        for nPoint in range(nold):
            t = (nPoint + 1) / m
            newPoly[nPoint + 1] = (oldPoly[nPoint]*((nPoint + 1)/nold - t) + 
                   oldPoly[nPoint + 1]*(t - nPoint/nold)) * nold
        newPoly[m] = oldPoly[-1]
        oldPoly = np.copy(newPoly)
    return newPoly

#polyA = np.array(([0, 0], [1, 1.5]))
#polyB = np.array(([2, 2], [3, 3.5]))
#
#fig = plt.figure()
#default_size = fig.get_size_inches()
#fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
#bezA = bezierPoly(polyA)
#bezB = bezierPoly(polyB)
#intPoly = interpolatePoly(polyA, polyB, 0, 4)
#bezIntP = bezierPoly(intPoly)
#""" Print smoothed controlP """
#plt.plot([x[0] for x in polyA], [x[1] for x in polyA], marker = "x")
#plt.plot([x[0] for x in polyB], [x[1] for x in polyB], marker = ".")
#plt.plot([x[0] for x in bezA], [x[1] for x in bezA], marker = "x")
#plt.plot([x[0] for x in bezB], [x[1] for x in bezB], marker = ".")
#""" Print interpolating polynomial """
#plt.plot([x[0] for x in intPoly], [x[1] for x in intPoly], marker = "x")
#plt.plot([x[0] for x in bezIntP], [x[1] for x in bezIntP], marker = ".")
#
#plt.show()


