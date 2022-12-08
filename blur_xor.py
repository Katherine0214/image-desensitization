from utils import *
from models import SRResNet
import time
from PIL import Image
from PIL import ImageFilter
import cv2
import xml.etree.ElementTree as ET
import os
from pathlib import Path


# 原图片、标签文件、裁剪图片、最终保存图片路径
ori_img_path = '/home/zhangc/project/image-desensitization/data/ori_plate'
xml_path = '/home/zhangc/project/image-desensitization/data/xml_plate'
obj_img_path = '/home/zhangc/project/image-desensitization/results/crop'
result_save_path = '/home/zhangc/project/image-desensitization/results'

# 模型参数
resize_base = 60        # 将原crop缩小的比例         【需要根据输入图片大小及ROI所占面积大小来调节】
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 8      # 放大比例
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
                ratio = (int(x1) - int(x0))/(int(y1) - int(y0))
                
                # 加载预训练模型
                if name == 'plate':
                    srresnet_checkpoint = "/home/zhangc/project/image-desensitization/model/checkpoint_srresnet_plate_8.pth"
                else:
                    srresnet_checkpoint = "/home/zhangc/project/image-desensitization/model/checkpoint_srresnet_face_8.pth"
                
                # 加载模型SRResNet
                checkpoint = torch.load(srresnet_checkpoint, map_location={'cuda:0': 'cuda:1'})  # 当用gpu=“1”训练并用“1”推理时，需要将map_location删掉
                srresnet = SRResNet(large_kernel_size=large_kernel_size,
                                    small_kernel_size=small_kernel_size,
                                    n_channels=n_channels,
                                    n_blocks=n_blocks,
                                    scaling_factor=scaling_factor)
                srresnet = srresnet.to(device)
                srresnet.load_state_dict(checkpoint['model'])
                srresnet.eval()
                model = srresnet
                
                # 加载图像
                ori_crop_img = Image.open(crop_imgPath, mode='r')
                ori_crop_img = ori_crop_img.convert('RGB')
                crop_img = Image.open(crop_imgPath, mode='r')
                crop_img = ori_crop_img.resize((int(resize_base * ratio), resize_base), Image.ANTIALIAS)   #  Image.NEAREST ：低质量 , Image.BILINEAR：双线性 , Image.BICUBIC ：三次样条插值 , Image.ANTIALIAS：高质量
                
                ######################## Encode: 将crop_img（ori_crop_img）编码 -- 先高斯滤波模糊处理，再用key编码 #############################
                crop_img = crop_img.filter(ImageFilter.GaussianBlur(radius=1))  
                p_crop_img = crop_img.load()
                key = Image.open('/home/zhangc/project/image-desensitization/data/key.jpg')
                key = key.resize((crop_img.width, crop_img.height), Image.ANTIALIAS)
                p_key = key.load()
                
                for i in range(key.width):
                    for j in range(key.height):
                        r1,g1,b1 = p_crop_img[i,j]
                        r2,g2,b2 = p_key[i,j]
                        p_crop_img[i,j] = (r1^r2, g1^g2, b1^b2)
                encode_crop_img = crop_img.resize((int(crop_img.width * scaling_factor),int(crop_img.height * scaling_factor)),Image.BICUBIC)
                
                ######################## Decode: 将crop_img（编码输出的crop_img）解码 -- 先用key解码，再用超分重建网络去模糊 #############################  
                crop_img_decode = crop_img
                p_crop_img_decode = crop_img_decode.load()             
                for i in range(key.width):
                    for j in range(key.height):
                        r1,g1,b1 = p_key[i,j]
                        r2,g2,b2 = p_crop_img_decode[i,j]
                        p_crop_img_decode[i,j] = (r1^r2, g1^g2, b1^b2)
                
                lr_crop_img = convert_image(crop_img_decode, source='pil', target='imagenet-norm')
                lr_crop_img.unsqueeze_(0)
                start = time.time()
                lr_crop_img = lr_crop_img.to(device)  # (1, 3, w, h ), imagenet-normed

                # 模型推理
                with torch.no_grad():
                    decode_crop_img = model(lr_crop_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
                    decode_crop_img = convert_image(decode_crop_img, source='[-1, 1]', target='pil')
                print('用时  {:.3f} 秒'.format(time.time()-start))
                
                
                #################### Step 3: 将 编码、解码后 的crop分别放回到原图中 #########################
                encode_crop_resize_img = encode_crop_img.resize((ori_crop_img.width,ori_crop_img.height),Image.BICUBIC)
                decode_crop_resize_img = decode_crop_img.resize((ori_crop_img.width,ori_crop_img.height),Image.BICUBIC)

                encode_img.paste(encode_crop_resize_img, (int(x0),int(y0),int(x1),int(y1)))
                decode_img.paste(decode_crop_resize_img, (int(x0),int(y0),int(x1),int(y1)))  
                
    encode_img.save(result_save_path + '/' + img_name  + '_blurxor_encode' + '.jpg') 
    decode_img.save(result_save_path + '/' + img_name  + '_blurxor_decode' + '.jpg')






