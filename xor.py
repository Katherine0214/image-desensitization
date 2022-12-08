from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import numpy as np


# 原图片、标签文件、裁剪图片、最终保存图片路径
ori_img_path = '/home/zhangc/project/image-desensitization/data/ori_plate'
xml_path = '/home/zhangc/project/image-desensitization/data/xml_plate'
obj_img_path = '/home/zhangc/project/image-desensitization/results/crop'
result_save_path = '/home/zhangc/project/image-desensitization/results'


#################### Step 1: 根据xml文件对原图中的标注位置进行裁剪 #########################
Numpic = {} 
for img_file in os.listdir(ori_img_path):
    if img_file[-4:] in ['.png', '.jpg']: 
        img_filename = os.path.join(ori_img_path, img_file)  
        img_cv = cv2.imread(img_filename) 
        
        img_name = (os.path.splitext(img_file)[0])  
        xml_name = xml_path + '/' + '%s.xml' % img_name 
        
        encode_img = Image.open(img_filename) 
        decode_img = Image.open(img_filename)  
        
        if os.path.exists(xml_name): 
            root = ET.parse(xml_name).getroot()  
            for obj in root.iter('object'):  
                name = obj.find('name').text  
                xmlbox = obj.find('bndbox') 
                x0 = xmlbox.find('xmin').text  
                y0 = xmlbox.find('ymin').text
                x1 = xmlbox.find('xmax').text
                y1 = xmlbox.find('ymax').text
                obj_img = img_cv[int(y0):int(y1), int(x0):int(x1)] 
                Numpic.setdefault(name, 0)  
                Numpic[name] += 1 
                my_file = Path(obj_img_path + '/' + name)  
                if 1 - my_file.is_dir():  
                    os.mkdir(obj_img_path + '/' + str(name))
                cv2.imwrite(obj_img_path + '/' + name + '/' + img_name + '_' + '%04d' % (Numpic[name]) + '.jpg', obj_img) 


                #################### Step 2: 对crop的图片进行：编码、解码 处理 #########################
                crop_imgPath = obj_img_path + '/' + name + '/' + img_name + '_' + '%04d' % (Numpic[name]) + '.jpg'
                ori_crop_img = cv2.imread(crop_imgPath)
                ori_crop_img=cv2.cvtColor(ori_crop_img,cv2.COLOR_BGR2RGB)
                key = cv2.imread('/home/zhangc/project/image-desensitization/data/key.jpg')
                key = cv2.resize(key, (ori_crop_img.shape[1],ori_crop_img.shape[0]))
                key2 = cv2.flip(key,1)
                # 编码
                encode_crop_img = np.bitwise_xor(ori_crop_img, key) 
                encode_crop_img = np.bitwise_xor(encode_crop_img, key2)       
                # 解码                    
                decode_crop_img = np.bitwise_xor(encode_crop_img, key2)
                decode_crop_img = np.bitwise_xor(decode_crop_img, key)

                #################### Step 3: 将 编码、解码后 的crop分别放回到原图中 #########################
                encode_img = np.asarray(encode_img)
                decode_img = np.asarray(decode_img)
                encode_img[int(y0):int(y1), int(x0):int(x1)] = encode_crop_img
                decode_img[int(y0):int(y1), int(x0):int(x1)] = decode_crop_img
    
                
    encode_img=cv2.cvtColor(encode_img,cv2.COLOR_BGR2RGB)     
    decode_img=cv2.cvtColor(decode_img,cv2.COLOR_BGR2RGB)     
      
    cv2.imwrite(result_save_path + '/' + img_name  + '_xor_encode' + '.jpg',encode_img) 
    cv2.imwrite(result_save_path + '/' + img_name  + '_xor_decode' + '.jpg',decode_img)





