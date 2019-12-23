from django.shortcuts import render
import base64
from PIL import Image
from . import STARGAN_BOTH
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
# from facedetect_project.Face_Super_Resolution_master.dlib_alignment import dlib_detect_face, face_recover
# import torch
# import torchvision.transforms as transforms
# from facedetect_project.Face_Super_Resolution_master.models.SRGAN_model import SRGANModel
# import argparse
# from facedetect_project.Face_Super_Resolution_master import utils


# image_dir은 14*1 사진의 경로및 파일이름.
def seperate(image_dir, result_dir):
    # 이미지 불러오기
    img = cv2.imread(image_dir)

    # 가로로 긴 이미지를 잘라서 담을 리스트
    img_list = []

    # 리스트에 간격이 일정하게 자름.
    for i in range(14):
        img_list.append(img[0:256, 256*i:256*(i+1)])
    for i in range(14):
        fin_image = img_list[i]
        cv2.imwrite(os.path.join(result_dir,
        os.path.split(image_dir)[-1].split('.')[0] + '_'+ str(i) + '.jpg'), fin_image)
    return 0;

def home(request):
    return render(request, 'home.html')


def check(request):
    if request.method == "POST":
        image_file = request.POST.get('image')
        data = image_file.split(',')[1]
        imgdata = base64.b64decode(data)
        filename = 'facedetect_project/static/img/image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata)
    image_file = Image.open(r'facedetect_project\static\img\image.jpg')
    image_file = image_file.convert('RGB')
    image_file.save(r'facedetect_project\static\img\image.jpg')
    image_dir = r'facedetect_project\static\img\image.jpg'
    result_dir = r'facedetect_project\static\img\result'
    #stargan
    STARGAN_BOTH.test_multi(image_dir,result_dir=result_dir, c_org1=[1,0,0,1,1],c_org2=[1,0,0,0,1,1,0,0])
    
    #cut
    seperate(r'C:\Users\user\Desktop\CNN\web\facedetect_project\facedetect_project\static\img\result\image.jpg',result_dir=r"C:\Users\user\Desktop\CNN\web\facedetect_project\facedetect_project\static\img\cut")

    #super
    # img_path = r"C:\Users\user\Desktop\CNN\web\facedetect_project\facedetect_project\static\img\cut"
    # origin_path = os.path.join(img_path, 'original')
    # list_photo = os.listdir(origin_path)\
    # for i in range(len(list_photo)):
    #     img = os.path.join(origin_path, list_photo[i])
    #     img = utils.read_cv2_img(img)
    #     output_img, rec_img = sr_forward(img)
    #     if i == 0 :
    #         re_img=rec_img
    #     else :
    #         re_img=np.concatenate((re_img,rec_img),axis=1)        
    # utils.save_image(re_img, os.path.join(img_path, 'resol.jpg'))

    #detect cut
    # resolution 직후 이미지 경로
    dir = r'C:\Users\user\Desktop\CNN\web\facedetect_project\facedetect_project\static\img\result\image.jpg'
    img = cv2.imread(dir)

    face_cascade = cv2.CascadeClassifier('C:/Users/user/Desktop/CNN/web/facedetect_project/facedetect_project/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/user/Desktop/CNN/web/facedetect_project/facedetect_project/haarcascades/haarcascade_eye.xml')

    dic={}
    for i in range(14):
        dic['img'+str(i)] = img[0:256, 256*i:256*(i+1)]

    img = dic['img0']
    # Resize the image to save space and be more manageable.
    # We do this by calculating the ratio of the new image to the old image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    w_size = 0
    h_size = 0
    #Perform the resizing and store the resized image in variable resized

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    #Identify the face and eye using the haar-based classifiers.
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = resized[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
    x, y, w, h = faces[0]

    if eyes.shape[0] <= 1 :
        w_size = 0
        h_size = 0
    else:
        w_size = int(2/5 * w)
        h_size = int(1/50 * h)
        
        p_pan=pd.DataFrame(eyes).iloc[:,1:]
        p_pan=pd.DataFrame(eyes).T
        p_pan=abs(p_pan.corr(method='pearson'))

        #print(p_pan)
        filter_pan = p_pan[(p_pan<1) & (p_pan>0.6)]
        cal_pan=filter_pan.sum() > 0.6
        cal_pan.index[cal_pan]
        
        #print(cal_pan)
        eyes=eyes[cal_pan.index[cal_pan],:]
        
        #y값을 비교해서 boundury 안에 있으면 x값이 가장작은 행선택
        if((eyes[0,1] > eyes[1,1]-5) | (eyes[0,1] < eyes[1,1] +5)):
            eyes = [eyes[np.argmin(eyes[:,0]),:]]
        
        #아니면 y값이 가장작은 행 선택.
        else:
            eyes=[eyes[np.argmin(eyes[:,1]),:]]

    ex, ey, ew, eh = eyes[0]
    for (ex,ey,ew,eh) in eyes :
        img=cv2.rectangle(roi_color,(ex-7,ey-7),(ex+ew+w_size,ey+eh+h_size),(0,0,0),0)
        #Display the bounding box for the face and eyes

        #print(eyes[0])
        #plt.imshow(img[ey-4 : ey+eh+h_size, ex-5: ex+ew+w_size])
        #cv2.imshow('img',resized)
        #cv2.waitKey(0)
        #cv2.imwrite(os.path.join(result_dir, os.path.split(directory)[-1]), img[ey-5 : ey+eh+h_size, ex-5: ex+ew+w_size])

    result_dir = r'C:\Users\user\Desktop\CNN\web\facedetect_project\facedetect_project\static\img\detectcut'
    
    eye_xy = []
    for i in range(14):
        cv2.imwrite(os.path.join(result_dir, os.path.split(dir.split('.')[0])[-1]+'_'+str(i)+'.jpg'),
        cv2.resize(dic['img'+str(i)],(500,500))[y+eh-7:y+eh+ey+h_size,x+ex-7:x+ex+ew+w_size])

    # 진영        
    # stargan에서 나오는 이미지 불러들이기 
    img  = cv2.imread("C:/Users/user/Desktop/CNN/web/facedetect_project/facedetect_project/static/img/result/image.jpg", cv2.IMREAD_COLOR)

    #stargan에서 나오는 이미지 14개 한꺼번에 붙여져있는거 하나씩으로 나눠줘서 img_list로 저장 
    # img_list[0]에는 원본이미지가 들어감 아현언니 원래사진 [1]부터 gan으로 생성된 사진 들어감
    # 밑에서 나오는 리스트의 0은 무조건 원본이미지정보 
    img_list = [] 

    for i in range(0, 3584, 256) :
        img_part = img[:,i:i+256]
        img_list.append(img_part)


    # img_list에서 eye detection한거 eye_list로 저장 
    eye_list = []
    for i in range(14) :
        img_address = "C:/Users/user/Desktop/CNN/web/facedetect_project/facedetect_project/static/img/detectcut/image_"+str(i)+".jpg"
        img_eye = cv2.imread(img_address, cv2.IMREAD_COLOR) # test할때는 임의로 눈부분 자름 
        eye_list.append(img_eye)


    hist_list = [] 
    eyeRGB_list = [] # 나중에 이미지 rgb 형태로 출력하기 위해 만든 리스트 

    for i in range(len(eye_list)) :
        img_rgb = cv2.cvtColor(eye_list[i], cv2.COLOR_BGR2RGB) # cv2는 RGB형식으로만 인식하기 때문에 RGB형태로 바꿈 
        eyeRGB_list.append(img_rgb) #rgb형식으로 저장 
        hist = cv2.calcHist(img_rgb, [0,1,2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_list.append(hist)

    results = {} # method하나에 생기는 결과 저장 딕셔너리 
    reverse = True # Correlation 출력결과 내림차순 

    for (k, hist) in enumerate(hist_list) :
        # 두 히스토그램 사이 거리 계산 
        d = cv2.compareHist(hist_list[0], hist, cv2.HISTCMP_CORREL) # 아현언니 원래사진[0]이랑 비교하는 함수 
        results[k] = d

    #상관계수 결과 내림차순으로 sort
    results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(eyeRGB_list[0]) #원본 이미지 출력 
    plt.axis("off")

    # results figure 그리기 
    fig = plt.figure("Result: %s" % ('Correlation'), figsize=(13,5))
    fig.suptitle('Correlation', fontsize = 20)

    # loop over the results
    for (i, (v, k)) in enumerate(results):
        # result 보여주기
        ax = fig.add_subplot(1, len(eyeRGB_list), i + 1)
        ax.set_title("%s: %.2f" % (k, v))# 상관관계수 출력 
        plt.imshow(eyeRGB_list[k])#원본 이미지랑 gan으로 생성된 이미지 출력
        plt.axis("off")


    plt.savefig('C:/Users/user/Desktop/CNN/web/facedetect_project/facedetect_project/static/img/cor_image.jpg')


    return render(request, 'check.html')



