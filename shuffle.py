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
                M,N,_ = ori_crop_img.shape
                
                # 编码
                r_ori = np.arange(M)
                r_shuffle = r_ori.copy()
                np.random.shuffle(r_shuffle)
                r_restore = np.argsort(r_shuffle)   # 将r_shuffle中的元素从小到大排列，提取其在排列前对应的index(索引)输出

                c_ori = np.arange(N)
                c_shuffle = c_ori.copy()
                np.random.shuffle(c_shuffle)
                c_restore = np.argsort(c_shuffle)

                img_r_emb = ori_crop_img.copy()[r_shuffle]
                img_c_emb = ori_crop_img.copy()[:,c_shuffle]
                encode_crop_img = img_r_emb.copy()[:,c_shuffle]
                
                # 解码
                decode_crop_img = encode_crop_img.copy()[:,c_restore]
                decode_crop_img = decode_crop_img[r_restore]

                #################### Step 3: 将 编码、解码后 的crop分别放回到原图中 #########################
                encode_img = np.asarray(encode_img)
                decode_img = np.asarray(decode_img)
                encode_img[int(y0):int(y1), int(x0):int(x1)] = encode_crop_img
                decode_img[int(y0):int(y1), int(x0):int(x1)] = decode_crop_img
    
                
    encode_img=cv2.cvtColor(encode_img,cv2.COLOR_BGR2RGB)     
    decode_img=cv2.cvtColor(decode_img,cv2.COLOR_BGR2RGB)     
      
    cv2.imwrite(result_save_path + '/' + img_name  + '_shuffle_encode' + '.jpg',encode_img) 
    cv2.imwrite(result_save_path + '/' + img_name  + '_shuffle_decode' + '.jpg',decode_img)





