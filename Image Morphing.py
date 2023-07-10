import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import imageio

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img1 = cv2.imread("img1.jpeg")
img4 = cv2.imread("img1.jpeg")
img1=cv2.resize(img1,(300,400),interpolation =cv2.INTER_NEAREST)
img4=cv2.resize(img4,(300,400),interpolation =cv2.INTER_NEAREST)
gray = cv2.cvtColor(src=img1,code=cv2.COLOR_BGR2GRAY)
arr1=[[0,0],[img1.shape[1],img1.shape[0]],[img1.shape[1],0],[0,img1.shape[0]]]
faces = detector(gray)
for face in faces:
    x1 = face.left() 
    y1 = face.top() 
    x2 = face.right() 
    y2 = face.bottom() 
    landmarks = predictor(image=gray, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        arr1.append((int(x),int(y)))



img2 = cv2.imread("img2.jpeg")
img3 = cv2.imread("img2.jpeg")
img2=cv2.resize(img2,(300,400),interpolation =cv2.INTER_NEAREST)
img3=cv2.resize(img3,(300,400),interpolation =cv2.INTER_NEAREST)
gray2 = cv2.cvtColor(src=img2, code=cv2.COLOR_BGR2GRAY)
arr2=[[0,0],[img2.shape[1],img2.shape[0]],[img2.shape[1],0],[0,img2.shape[0]]]
faces = detector(gray2)
for face in faces:
    x1 = face.left() 
    y1 = face.top() 
    x2 = face.right() 
    y2 = face.bottom() 
    landmarks = predictor(image=gray2, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        arr2.append((int(x),int(y)))


points=np.array(arr1)
tri = Delaunay(points)
arr_d=tri.simplices


def fun(alpha,img1,img2,arr1,arr2,arr_d):
    arr5=[]
    newimg1=np.zeros(img2.shape,dtype=np.uint8)
    for i in range(len(arr1)):
        arr5.append((int(((1-alpha)*arr1[i][0]+alpha*arr2[i][0])),int((1-alpha)*arr1[i][1]+alpha*arr2[i][1])))
    new_img=np.zeros(img1.shape)
    
    for i in range(len(arr_d)):
        a=arr_d[i][0]
        b=arr_d[i][1]
        c=arr_d[i][2]
        pts1=np.float32([arr1[a],arr1[b],arr1[c]])
        pts=np.float32([arr5[a],arr5[b],arr5[c]])
        pts2=np.float32([arr2[a],arr2[b],arr2[c]])


        contours = np.array([arr5[a],arr5[b],arr5[c]])
        img_temp = np.zeros((img1.shape[0],img1.shape[1],3) ) 
        cv2.fillPoly(img_temp, pts =[contours], color=(1,1,1))
        
        img_1=np.zeros(img1.shape,dtype=np.uint8)
        img_2=np.zeros(img2.shape,dtype=np.uint8)
        
        new_img1=np.zeros(img1.shape,dtype=np.uint8)
        new_img2=np.zeros(img2.shape,dtype=np.uint8)
        M = cv2.getAffineTransform(pts1, pts)
        new_img1= cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
        M2 = cv2.getAffineTransform(pts2, pts)
        new_img2= cv2.warpAffine(img2, M2, (img2.shape[1], img2.shape[0]))  
        
        img_temp=img_temp.astype("uint8")
        n_img=((100-100*alpha)*new_img1+100*alpha*new_img2)/100
        n_img=n_img.astype("uint8")
        newimg1=newimg1*(1-img_temp)+(n_img*img_temp)
        newimg1=newimg1.astype("uint8")

    return(newimg1[...,::-1])


def usinginbuilt(numb):
    giff=[]
    ind=0.0
    number=numb
    while(ind<=1):
        giff.append(fun(ind,img1,img2,arr1,arr2,arr_d))
        ind=(number*ind+1)/number
    return giff

def usinginputtext(numb):
    array1=[[0,0],[img1.shape[1],img1.shape[0]],[img1.shape[1],0],[0,img1.shape[0]]]
    array2=[[0,0],[img2.shape[1],img2.shape[0]],[img2.shape[1],0],[0,img2.shape[0]]]
    with open('input.txt','r') as file:    
        for line in file:
            i=0       
            for word in line.split():
                if(i==0 or i==2):
                    temp=word
                if(i==1):
                    array1.append([int(temp),int(word)])
                if(i==3):
                    array2.append([int(temp),int(word)])
                i=i+1
    points=np.array(array1)
    trian = Delaunay(points)
    array_d=trian.simplices

    giff1=[]
    ind1=0.0
    number1=numb
    while(ind1<=1):
        giff1.append(fun(ind1,img1,img2,array1,array2,array_d))
        ind1=(number1*ind1+1)/number1
    return giff1


giff=usinginputtext(100)
with imageio.get_writer("morphingusinginputfile.gif",mode="I") as writer:
    for idx,frame in enumerate(giff):
        writer.append_data(frame)

giff=usinginbuilt(100)
with imageio.get_writer("morphinginbuilt.gif",mode="I") as writer:
    for idx,frame in enumerate(giff):
        writer.append_data(frame)

