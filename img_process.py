from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image,ImageEnhance
import PIL
import scipy.misc
import time
import skimage
from skimage.transform import resize
import skimage.color

###################################################

#######################正規化使單獨每張圖片均值為0標準差為1(輸入:全部圖片)//example-wise normalization
def standard(x):
	x = x.astype('float')
	for i in range(x.shape[0]):
		for j in range(3):
			mean=np.mean(x[i][...,j])
			std=np.std(x[i][...,j])
			x[i][...,j]=(x[i][...,j]-mean)/std
	return x

#######################將圖片水平翻轉(輸入:全部圖片)#######################
def horizontal(x):
	temp=[]
	for i in range(x.shape[0]):
		img = scipy.misc.toimage(x[i,:])
		img=img.transpose(Image.FLIP_LEFT_RIGHT)
		img=np.asarray(img)
		temp.append(img)
	temp=np.asarray(temp)
	return temp
	
#######################增強清晰度再增強對比度#######################    
def sharp_contrast(image,sharpness=2,contrast=2):#輸入(圖像,清晰度質,對比度質)
	for i in range(image.shape[0]):
		img = scipy.misc.toimage(image[i,:])
		enhancer = ImageEnhance.Sharpness(img)
		img=enhancer.enhance(sharpness)
		enhancer = ImageEnhance.Contrast(img)
		img=enhancer.enhance(contrast)
		image[i,:]=img
	return image
	
	
#######################增強彩度再增強清晰度#######################   
def color_sharp(image,color=2,sharpness=2):#輸入(圖像,彩度質,清晰度質)
	for i in range(image.shape[0]):
		img = scipy.misc.toimage(image[i,:])
		enhancer = ImageEnhance.Color(img)
		img=enhancer.enhance(color)
		enhancer = ImageEnhance.Sharpness(img)
		img=enhancer.enhance(sharpness)
		image[i,:]=img
	return image

#######################裁切圖像#######################       
def crop_img(image,size):#(圖像,裁切後邊長)
	resize=int((image.shape[1]-size)/2)
	image=image[:,resize:(image.shape[1]-resize),resize:(image.shape[1]-resize),:]
	return image

#######################增大圖像(邊緣補0)####################### 
def increase_img(img,to_size=32):#(圖像,增加後邊長)
	resize=int((to_size-img.shape[1])/2)
	white_img=np.zeros((img.shape[0],to_size,to_size,img.shape[3]))
	white_img[:,resize:(to_size-resize),resize:(to_size-resize),:]+=img
	return white_img

#######################每筆資料距離最近的圖像#######################
def nearest_img(target,image,dist_num=0):#輸入(資料集A,資料集B,使用距離:0歐式/1曼哈頓),輸出:與資料集A筆數相同的資料,為資料集B中與資料集A距離最近者
		
	nearest=[]
	start_time = time.time() 
	dist_name=['eu','manh']
	print('比較資料數','A:',target.shape[0],'B:',image.shape[0],'使用距離:',dist_name[dist_num])
	for i in range(target.shape[0]):
		dist=[]	
		for j in range(image.shape[0]):
			if dist_num==0:dist_temp=np.sqrt(np.sum(np.power(target[i,:]-image[j],2)))
			elif dist_num==1:dist_temp=np.sum(np.absolute(target[i]-image[j]))                
			dist.append(dist_temp)
		dist=np.asarray(dist)
		nearest.append(image[np.argmin(dist)])
	nearest=np.asarray(nearest)
	print('搜尋完成,花費時間:',time.time()-start_time)
	return nearest
	
#######################各類別取資料,筆數與類別數相同#######################
def on_class_each_row(data):#(輸入:多維陣列,每維一類資料 ex:data[0]~data[9]各存一類共10類)
	len_size=int(data.shape[0])
	img_size=int(data[0].shape[1])
	img_channel=int(data[0].shape[3])
	final=np.zeros((1,img_size,img_size,img_channel))
	for i in range(len_size):
		final=np.concatenate((final,data[i][0:len_size]))
	return final[1:]
	
	
#######################改變資料範圍大小#######################
def re_scale(data,new_min=0,new_max=1):
	old_range=np.max(data)-np.min(data)
	old_min=np.min(data)
	new_range=new_max-new_min
	new_data=((data-old_min)*new_range)/old_range+new_min
	return new_data
	
#######################將一張圖顏色分為r,g,b三張圖#######################
def r_g_b(img):
    img_r=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    img_r[...,0]=img[...,0]
    img_g=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    img_g[...,1]=img[...,1]
    img_b=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    img_b[...,2]=img[...,2]
    return img_r,img_g,img_b
  
  
#######################區分 feature maps 每層#######################    
def fm_as_pic(feature_maps):
    img=np.zeros((1,feature_maps.shape[1],feature_maps.shape[1]))
    for i in range(feature_maps.shape[3]):
        img=np.concatenate((img,feature_maps[:,:,:,i]),axis=0)
    img=np.reshape(img,[img.shape[0],img.shape[1],img.shape[2],1])
    return img[1:]

#######################挑選最近鄰為同類的資料#######################
def choose_nearest(target,data_class,image,label,dist_num=0):#輸入(資料集A,資料集B,使用距離:0歐式/1曼哈頓),輸出:與資料集A筆數相同的資料,為資料集B中與資料集A距離最近者		
	nearest=[]
	start_time = time.time() 
	dist_name=['eu','manh']
	print('比較資料數','A:',target.shape[0],'B:',image.shape[0],'使用距離:',dist_name[dist_num])
	for i in range(target.shape[0]):
		dist=[]	
		for j in range(image.shape[0]):
			if dist_num==0:dist_temp=np.sqrt(np.sum(np.power(target[i,:]-image[j],2)))
			elif dist_num==1:dist_temp=np.sum(np.absolute(target[i]-image[j]))                
			dist.append(dist_temp)
		dist=np.asarray(dist)
		if ((np.argmax(label[np.argmin(dist)]))==(data_class)):
			nearest.append(target[i])
	nearest=np.asarray(nearest)
	print('搜尋完成,花費時間:',time.time()-start_time)
	return nearest
#######################圖片隨機剪裁#######################
def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

#######################圖片放大#######################
def img_resize(x,s):
	temp=[]
	for i in range(x.shape[0]):
		img = resize(x[i], [s,s,1])		
		img=np.asarray(img)
		temp.append(img)
	temp=np.asarray(temp)
	return temp

#######################Image to binary#######################
def i2b(x):
	img=[]
	for i in range(len(x)):
		ret,thresh = cv2.threshold(x[i],0.5,1,cv2.THRESH_BINARY)
		img.append(thresh)
	img = np.asarray(img)
	img = np.expand_dims(img, axis=3)
	return img



def cifar2mnistform(x):
	img=[]
	for i in range(len(x)):
		temp = skimage.color.rgb2gray(x[i,:,:,:])
		img.append(temp)
	img = np.asarray(img)
	img = img[:,2:30, 2:30]
	return img