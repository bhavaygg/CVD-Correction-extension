#For API
from flask import Flask, request, jsonify
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS

#For image processing
import cv2
import os
import numpy as np
from PIL import Image
import requests

Directory = "E:/Project/Information_Retrival/Project/Extension/images"

app = Flask(__name__)
api = Api(app)
CORS(app)

# CORS Headers 
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,true')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

parser = reqparse.RequestParser()
parser.add_argument('sequence', type=int, required = True, help = 'Sequence number')
parser.add_argument('link', type=str, required = True, help = 'Link of image')
parser.add_argument('cvd_type', type=str, required = True, help = 'CVD type')


def CVD_Stim (img, CVD_type, simple_linear_transform=False):
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

class Test(Resource):
    def __init__(self) -> None:
        self.__sequence = parser.parse_args().get('sequence', "None")
        self.__link = parser.parse_args().get('link', "None")
        self.__cvd_type = parser.parse_args().get('cvd_type', "None")

    def get(self):  
        try:
            image = np.array(Image.open(requests.get(self.__link, stream=True).raw))
            if self.__cvd_type=="proto":
                proto_corrected = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia", simple_linear_transform = True)), cv2.COLOR_BGR2LAB)
                cv2.imwrite(f"{self.__sequence}.jpg", np.uint8(cv2.cvtColor(np.uint8(proto_corrected), cv2.COLOR_LAB2BGR)))
            elif self.__cvd_type=="deuto":
                deuto_corrected = cv2.cvtColor(np.uint8(CVD_Stim(image, "Deuteranope", simple_linear_transform = True)), cv2.COLOR_BGR2LAB)
                cv2.imwrite(f"{self.__sequence}.jpg", np.uint8(cv2.cvtColor(np.uint8(deuto_corrected), cv2.COLOR_LAB2BGR)))
            elif self.__cvd_type=="trita":
                tritano_corrected = cv2.cvtColor(np.uint8(CVD_Stim(image, "Tritanopia", simple_linear_transform = True)), cv2.COLOR_BGR2LAB)
                cv2.imwrite(f"{self.__sequence}.jpg", np.uint8(cv2.cvtColor(np.uint8(tritano_corrected), cv2.COLOR_LAB2BGR)))
            else:
                print("Invalid CVD type")
            Image_json = {"path": f"{self.__sequence}.jpg", "response": 200}
            return jsonify(Image_json)
        except:
            print("Invalid Link")
            return jsonify({"path": self.__link, "response": 404})

    def remove_jpg(self):
        filelist = [ f for f in os.listdir(Directory) if f.endswith(".jpg") ]
        for file in filelist:
            os.remove(os.path.join(Directory, file))

api.add_resource(Test, '/process/')


if __name__ == "__main__":
    app.run(debug=True)