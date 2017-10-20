# #!python
# from numpy import *
# from numpy.random import *
#
#
# def resample(weights):
#   n = len(weights)
#   indices = []
#   C = [0.] + [sum(weights[:i+1]) for i in range(n)]
#   u0, j = random(), 0
#   for u in [(u0+i)/n for i in range(n)]:
#     while u > C[j]:
#       j+=1
#     indices.append(j-1)
#   return indices
#
#
# def particlefilter(sequence, pos, stepsize, n):
#   seq = iter(sequence)
#   x = ones((n, 2), int) * pos                   # Initial position
#   f0 = seq.next()[tuple(pos)] * ones(n)         # Target colour model
#   yield pos, x, ones(n)/n                       # Return expected position, particles and weights
#   for im in seq:
#     np.add(x, uniform(-stepsize, stepsize, x.shape), out=x, casting="unsafe")  # Particle motion model: uniform step
#     x  = x.clip(zeros(2), array(im.shape)-1).astype(int) # Clip out-of-bounds particles
#     f  = im[tuple(x.T)]                         # Measure particle colours
#     w  = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
#     w /= sum(w)                                 # Normalize w
#     yield sum(x.T*w, axis=1), x, w              # Return expected position, particles and weights
#     if 1./sum(w**2) < n/2.:                     # If particle cloud degenerate:
#       x  = x[resample(w),:]                     # Resample particles according to weights
#
#
# #!python
# if __name__ == "__main__":
#   from pylab import *
#   from itertools import izip
#   import time
#   from IPython import display
#
#   ion()
#   seq = [ im for im in zeros((20,240,320), int)]      # Create an image sequence of 20 frames long
#   x0 = array([120, 160])                              # Add a square with starting position x0 moving along trajectory xs
#   xs = vstack((arange(20)*3, arange(20)*2)).T + x0
#   for t, x in enumerate(xs):
#     xslice = slice(x[0]-8, x[0]+8)
#     yslice = slice(x[1]-8, x[1]+8)
#     seq[t][xslice, yslice] = 255
#
#   for im, p in izip(seq, particlefilter(seq, x0, 8, 100)): # Track the square through the sequence
#     pos, xs, ws = p
#     position_overlay = zeros_like(im)
#     position_overlay[np.array(pos).astype(int)] = 1
#     particle_overlay = zeros_like(im)
#     particle_overlay[tuple(xs.T)] = 1
#     draw()
#     time.sleep(0.3)
#     clf()                                           # Causes flickering, but without the spy plots aren't overwritten
#     imshow(im,cmap=cm.gray)                         # Plot the image
#     spy(position_overlay, marker='.', color='b')    # Plot the expected position
#     spy(particle_overlay, marker=',', color='r')    # Plot the particles
#     display.clear_output(wait=True)
#     display.display(show())
# -*- coding:utf-8 -*-

import random
import numpy as np
from scipy import stats
import cv
import cv2

class Image:
    def __init__(self):
        self.capture = cv.CreateCameraCapture(0)
        #captureのサイズを変更(プレゼン用)
        cv.SetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_WIDTH,512)
        cv.SetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_HEIGHT,512)
        self.image = cv.QueryFrame(self.capture)
        cv.ShowImage("Capture",self.image)
        self.size = (self.image.width,self.image.height)

    def create(self):
        self.image = cv.QueryFrame(self.capture)

    def getCol(self,sv):
        #!!!座標値をintに変換している
        x = sv[0]
        y = sv[1]
        #パーティクルが画面外にはみ出した場合は黒として返す
        if((x<0 or x>self.size[0]) or (y<0 or y>self.size[1])):
            return((0,0,0,0))
        else:
            return(cv.Get2D(self.image,int(sv[1]),int(sv[0])))

class SystemModel:
    def __init__(self,model):
        self.model = model
    def generate(self,sv,w):
        return(self.model(sv,w))

class Likelihood:
    def __init__(self,model):
        self.model = model

    def generate(self,sv,mv):
        return(self.model(sv,mv))

    def normalization(self,svs,mv):
        return(sum([self.generate(sv,mv) for sv in svs]))

def model_s(sv,w):
    #等速直線運動モデル
    #状態ベクトルは(x,y,vx,vy)を仮定
    F = np.matrix([[1,0,1,0],
                   [0,1,0,1],
                   [0,0,1,0],
                   [0,0,0,1]])
    return(np.array(np.dot(F,sv))[0]+w)

def model_l(sv,mv):
    #mv(cvイメージ)からsvの座標値(第1,2成分)に
    #対応する点の色情報を取得する
    mv_col = img.getCol(sv)
    mv_col = mv_col[0:3]
    target_col = (150,90,40)

    #尤度は色情報と指定する色の差に対するガウスモデル
    delta = np.array(mv_col)-np.array(target_col)
    dist_sqr = sum(delta*delta)
    sigma = 10000.0
    gauss = np.exp(-0.5*dist_sqr/(sigma*sigma)) / (np.sqrt(2*np.pi)*sigma)
    return(gauss)

def resampling(svs,weights):
    N = len(svs)
    #重みの大きい順にパーティクルをソート
    sorted_particle = sorted([list(x) for x in zip(svs,weights)],key=lambda x:x[1],reverse=True)
    #重みに従ってパーティクルをリサンプリング
    resampled_particle = []
    while(len(resampled_particle)<N):
        for sp in sorted_particle:
            resampled_particle += [sp[0]]*(sp[1]*N)
    resampled_particle = resampled_particle[0:N]

    return(resampled_particle)

def filtration(svs,mv,systemModel,likelihood):
    #システムモデルに用いる乱数を生成
    dim = len(svs[1])
    N = len(svs)
    sigma = 2.0 #スケール？
    rnorm = stats.norm.rvs(0,sigma,size=N*dim)
    ranges = zip([N*i for i in range(dim)],[N*i for i in (range(dim+1)[1:])])
    ws = np.array([rnorm[p:q] for p,q in ranges])
    ws = ws.transpose()

    #予測サンプルを生成
    svs_predict = [systemModel.generate(sv,w) for sv,w in zip(svs,ws)]

    #尤度重みを計算
    normalization_factor = likelihood.normalization(svs_predict,mv)
    likelihood_weights = [likelihood.generate(sv,mv)/normalization_factor for sv in svs_predict]
    #重みによってリサンプリング
    svs_resampled = resampling(svs_predict,likelihood_weights)
    return(svs_resampled)

#初期パーティクルを生成する。モデルが(x,y,vx,vy)の4次元モデルであることを仮定
def initStateVectors(imageSize,sampleSize):
    xs = [random.uniform(0,imageSize[0]) for i in range(sampleSize)]
    ys = [random.uniform(0,imageSize[1]) for i in range(sampleSize)]
    vxs = [random.uniform(0,5) for i in range(sampleSize)]
    vys = [random.uniform(0,5) for i in range(sampleSize)]

    return([list(s) for s in zip(xs,ys,vxs,vys)])

#パーティクル付きの画像を表示する
def showImage(svs,img):
    #パーティクル描画用のコピー
    dst = cv.CloneImage(img.image)
    #パーティクルを書き込み
    for sv in svs:
        #!!!パーティクル位置をintに変換している
        cv.Circle(dst,(int(sv[0]),int(sv[1])),3,cv.CV_RGB(0,0,255))
    #表示
    cv.Flip(dst,flipMode=1)
    cv.ShowImage("Capture",dst)
    #録画
    #cv2.WriteFrame(vw,dst)


if(__name__=="__main__"):
    #イメージソースを指定
    img = Image()
    #ビデオ録画
    #vw = cv2.VideoWriter('cap.avi', cv.CV_FOURCC(*'DIB '), 30.0, (512, 512), False)
    #パーティクル数を指定
    sampleSize = 100
    #モデルオブジェクトを生成
    systemModel = SystemModel(model_s)
    likelihood = Likelihood(model_l)
    #初期パーティクルを生成
    svs = initStateVectors(img.size,sampleSize)

    while(True):
        #描画
        showImage(svs,img)
        #観測
        img.create()
        #フィルタ
        svs = filtration(svs,img,systemModel,likelihood)
