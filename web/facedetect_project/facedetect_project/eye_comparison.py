#!/usr/bin/env python
# coding: utf-8



import cv2
from matplotlib import pyplot as plt



# stargan에서 나오는 이미지 불러들이기 
img  = cv2.imread("eye_comparison.jpg", cv2.IMREAD_COLOR)



#stargan에서 나오는 이미지 14개 한꺼번에 붙여져있는거 하나씩으로 나눠줘서 img_list로 저장 
# img_list[0]에는 원본이미지가 들어감 아현언니 원래사진 [1]부터 gan으로 생성된 사진 들어감
# 밑에서 나오는 리스트의 0은 무조건 원본이미지정보 
img_list = [] 

for i in range(0, 3584, 256) :
    img_part = img[:,i:i+256]
    img_list.append(img_part)



# img_list에서 eye detection한거 eye_list로 저장 
eye_list = []

for i in range(len(img_list)) :
    img_eye = img_list[i][100:120, 70:170] # test할때는 임의로 눈부분 자름 
    eye_list.append(img_eye)



#히스토그램 분포 출력하는 코드 
# eye detection한거 히스토그램 분포 만들어서 hist_list에  저장 
hist_list = [] 
eyeRGB_list = [] # 나중에 이미지 rgb 형태로 출력하기 위해 만든 리스트 

for i in range(len(eye_list)) :
    img_rgb = cv2.cvtColor(eye_list[i], cv2.COLOR_BGR2RGB) # cv2는 RGB형식으로만 인식하기 때문에 RGB형태로 바꿈 
    eye_RGB_list.append(img_rgb) #rgb형식으로 저장 
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


plt.show()

