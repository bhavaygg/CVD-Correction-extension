#!/usr/bin/env python
# coding: utf-8

# # Header files

# In[1]:


import cv2
import numpy as np
import gradio as gr


# # Functions

# In[2]:


def CVD_Stim (img, CVD_type, simple_linear_transform=True):
    '''
    Function to generate CVD simulation and correct them using image transformation
    Args:
        - img: Image to be simulated/corrected.
        - CVD_type: Type of CVD to be simulated/corrected. Options -> (Protanopia, Deuteranopia, Tritanopia).
        - simple_linear_transform: If True, image is corrected for the selected CVD instead of simulating the selected CVD.
    '''
    img = np.array(img)
    sizeImg = img.shape 
    if(len(sizeImg)==3):
        imgHeight = sizeImg[0]
        imgWidth  = sizeImg[1]
        imgB = img[:,:,0]
        imgG = img[:,:,1]
        imgR = img[:,:,2]
    else:
        imgHeight = 1
        imgWidth  = sizeImg[0]
        imgB = img[:,0]
        imgG = img[:,1]
        imgR = img[:,2]
    GAMMA  = 2.2
    imgRGBVec = np.concatenate(([imgR.flatten()], [imgG.flatten()], [imgB.flatten()]), axis = 0)
    imgRGBVec = np.power(imgRGBVec, GAMMA)

    rgb2lms = [[17.8824, 43.5161, 4.11935],[3.45565, 27.1554, 3.86714], [0.0299566, 0.184309, 1.46709]]
    lms2rgb = [[0.0809, -0.1305, 0.1167], [-0.0102, 0.0540, -0.1136], [-0.0004, -0.0041, 0.6935]]
    imgLMSVec = np.mat(rgb2lms) * np.mat(imgRGBVec)

    T = []
    if CVD_type == "Protanopia":
        T = [[0, 2.02344, -2.52581], [0, 1, 0] ,[0, 0, 1]] 
    elif CVD_type == "Deuteranope":
        T = [[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]
    else:
        T = [[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]

    imgSimLMS = T * imgLMSVec
    imgSimRGBVec = lms2rgb*imgSimLMS

    if simple_linear_transform == True:
        transform_matrix = [[1, 0, 0], [0.7, 1, 0], [0.7, 0, 1]]
        imgSimRGBVec = imgRGBVec + transform_matrix * (imgRGBVec - imgSimRGBVec)

    imgSimR = imgSimRGBVec[0,:]
    imgSimG = imgSimRGBVec[1,:]
    imgSimB = imgSimRGBVec[2,:]

    imgSimR = np.array(imgSimR, dtype = np.complex)
    imgSimG = np.array(imgSimG, dtype = np.complex)
    imgSimB = np.array(imgSimB, dtype = np.complex)


    imgSimR = np.real(np.power(imgSimR, 1/GAMMA))
    imgSimG = np.real(np.power(imgSimG, 1/GAMMA))
    imgSimB = np.real(np.power(imgSimB, 1/GAMMA))

    imgSimR = np.reshape(imgSimR, [imgHeight, imgWidth])
    imgSimG = np.reshape(imgSimG, [imgHeight, imgWidth])
    imgSimB = np.reshape(imgSimB, [imgHeight, imgWidth])

    imgSim =  cv2.merge((imgSimB,imgSimG,imgSimR))
    return imgSim


# # Driver

# In[3]:


def process_Image(image_file, size=200):
    image_Size = size
    image = np.array(cv2.imread(f"{image_file}"))
    cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    proto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia")), cv2.COLOR_BGR2LAB)
    deuto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Deuteranope")), cv2.COLOR_BGR2LAB)
    tritano = cv2.cvtColor(np.uint8(CVD_Stim(image, "Tritanopia")), cv2.COLOR_BGR2LAB)
    cv2.imwrite(f"{image_file.split('.')[0]}_protonopia.jpg", proto)
    cv2.imwrite(f"{image_file.split('.')[0]}_deuteranope.jpg", deuto)
    cv2.imwrite(f"{image_file.split('.')[0]}_tritanopia.jpg", tritano)


# In[4]:


# process_Image("s.jpg")


# In[7]:


def gradio_process_Image(image, cvd_type):
    cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if cvd_type=="Protanopia":
        proto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia")), cv2.COLOR_BGR2LAB)
        return proto
    elif cvd_type=="Deuteranope":
        deuto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Deuteranope")), cv2.COLOR_BGR2LAB)
        return deuto
    else:
        tritano = cv2.cvtColor(np.uint8(CVD_Stim(image, "Tritanopia")), cv2.COLOR_BGR2LAB)
        return tritano


# In[8]:


# image = np.array(cv2.imread("s.jpg"))
# gradio_process_Image(image)

# In[ ]:

print("Opening gradio")
iface = gr.Interface(	fn = gradio_process_Image, 
                        inputs = [gr.inputs.Image(shape=(1000, 1000)), gr.inputs.Radio(["Protanopia", "Deuteranope", "Tritanopia"])], 
                        outputs = ["image"])
iface.launch(share=True)

