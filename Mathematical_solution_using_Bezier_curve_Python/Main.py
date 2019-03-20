import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import BezFunc as bz
import time

"""

"""


im = Image.open("Fig_hidden_Nodes_3_level_1_ID_2.bmp")

"""
Este bloque es para cargar las dos partes visibles de la figura
"""
w, h, npImg_left = bz.img2Pixel(im,"left")
imPixelList_left = bz.getPxList(npImg_left)

w, h, npImg_right = bz.img2Pixel(im,"right")
imPixelList_right = bz.getPxList(npImg_right)

plt.plot([x[0] for x in imPixelList_left], [x[1] for x in imPixelList_left],linestyle="",marker = ".")
plt.plot([x[0] for x in imPixelList_right], [x[1] for x in imPixelList_right],linestyle="",marker = ".")
plt.show()

npPixelMap_left = bz.avgPixelMap(npImg_left,w,h)
npPixelMapList_left = bz.getPxList(npPixelMap_left)

npPixelMap_right = bz.avgPixelMap(npImg_right,w,h)
npPixelMapList_right = bz.getPxList(npPixelMap_right)

plt.plot([x[0] for x in npPixelMapList_left], [x[1] for x in npPixelMapList_left],linestyle="",marker = ".")
plt.plot([x[0] for x in npPixelMapList_right], [x[1] for x in npPixelMapList_right],linestyle="",marker = ".")

start_left,end_left = bz.findExtremePts(npPixelMapList_left,npPixelMap_left)
start_right,end_right = bz.findExtremePts(npPixelMapList_right,npPixelMap_right)

plt.plot(start_left[0],start_left[1],marker = "x")
plt.plot(end_left[0],end_left[1],marker = "x")
plt.plot(start_right[0],start_right[1],marker = "x")
plt.plot(end_right[0],end_right[1],marker = "x")
plt.show()

"""
Ahora CompareCurves devuelve los control points, los dos ultimos numeros que aparecen (20,4) se refieren al parametro
alpha (20) y al count_max(4). Alpha sirve para regular el grado de convergencia, cuanto mas alto mas exploratorio,
luego count_max es para darle oportunidades al algoritmo si encuentra un set de control points malos.
Antes, si encontraba una curva peor que la inicial, devolvia la anterior y acababa el fiteo, ahora si encuentra una peor, parte de
la anterior y disminuye el valor de alpha. Si supera el numero de oportunidades (en este caso 4), termina devolviendo la curva
anterior a la mal fiteada.
"""

#import imageio
#
#def plot_for_offset(limit):
#    # Data for plotting
#
#    fig, ax = plt.subplots(figsize=(8,8))
#        
#    A = np.array([[0,0], [1,1], [2,0]])
#
#    A = subDividePoly(A, limit+1)
#    bezA = bezierPoly(A)
#    
#    ax.plot([x[0] for x in A], [x[1] for x in A] , linestyle=":", linewidth="1", marker="o", color="k")
#    ax.plot([x[0] for x in bezA], [x[1] for x in bezA] , linewidth="2", marker="", color="g")
#
#    ax.grid()
#    ax.set(xlabel='', ylabel='', title=f'')
#
#    # IMPORTANT ANIMATION CODE HERE
#    # Used to keep the limits constant
#    ax.set_ylim(0, 1)
#    ax.set_xlim(0, 2)
#
#    # Used to return the plot as an image rray
#    fig.canvas.draw()       # draw the canvas, cache the renderer
#    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#    return image
#
#kwargs_write = {'fps':1.0, 'quantizer':'nq'}
#imageio.mimsave('./Bezier_2b.gif', [plot_for_offset(i) for i in range(50)], fps=5)

print("start New Figure")


import imageio

imgArray = []

#fig = plt.figure()

#ax.plot([x[0] for x in npPixelMapList_left], [x[1] for x in npPixelMapList_left],linestyle="",marker = ".")
#ax.plot([x[0] for x in npPixelMapList_right], [x[1] for x in npPixelMapList_right],linestyle="",marker = ".")

tstart = time.perf_counter()
controlPoly_left = bz.CompareCurves(10,start_left,end_left,w,h,imPixelList_left,npImg_left,50,10, npPixelMapList_left, npPixelMapList_right, imgArray)
controlPoly_right = bz.CompareCurves(10,start_right,end_right,w,h,imPixelList_right,npImg_right,50,10, npPixelMapList_left, npPixelMapList_right, imgArray)
print(time.perf_counter()-tstart)

kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./Test.gif', [image for i, image in enumerate(imgArray)], fps=1)

ctrPL = bz.subDividePoly(controlPoly_left, 2)
ctrPR = bz.subDividePoly(controlPoly_right, 2)
intPoly1, intPoly2 = bz.closeFigure(ctrPL, ctrPR, 0, 2)

bezListInt1 = bz.bezierPoly(intPoly1)
bezListInt2 = bz.bezierPoly(intPoly2)
bezCPL = bz.bezierPoly(ctrPL)
bezCPR = bz.bezierPoly(ctrPR)
""" Print smoothed controlP """
plt.plot([x[0] for x in ctrPL], [x[1] for x in ctrPL], marker = "x")
plt.plot([x[0] for x in bezCPL], [x[1] for x in bezCPL], marker = "")
plt.plot([x[0] for x in ctrPR], [x[1] for x in ctrPR], marker = "x")
plt.plot([x[0] for x in bezCPR], [x[1] for x in bezCPR], marker = "")
""" Print interpolating polynomial """
plt.plot([x[0] for x in intPoly1], [x[1] for x in intPoly1], marker = "o")
plt.plot([x[0] for x in bezListInt1], [x[1] for x in bezListInt1], marker = "")
plt.plot([x[0] for x in intPoly2], [x[1] for x in intPoly2], marker = "o")
plt.plot([x[0] for x in bezListInt2], [x[1] for x in bezListInt2], marker = "")



#plt.savefig('OutImageA.pdf', bbox_inches='tight')

fig = plt.figure()
default_size = fig.get_size_inches()
fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
plt.plot([x[0] for x in npPixelMapList_left], [250-x[1] for x in npPixelMapList_left],linestyle="",marker = ".")
plt.plot([x[0] for x in npPixelMapList_right], [250-x[1] for x in npPixelMapList_right],linestyle="",marker = ".")
""" Print smoothed controlP """

""" Print interpolating polynomial """

plt.plot([x[0] for x in bezListInt1], [250-x[1] for x in bezListInt1], marker = "")
plt.plot([x[0] for x in bezListInt2], [250-x[1] for x in bezListInt2], marker = "")

plt.savefig('OutImageB.pdf', bbox_inches='tight')

