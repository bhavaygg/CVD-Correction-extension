import cv2
import numpy as np
from pytube import YouTube
from pytube.cli import on_progress
import cv2
import numpy as np
import glob
from pathlib import Path
import os
import time
import ffmpeg
import matplotlib.pyplot as plt

def CVD_Stim (img, CVD_type, simple_linear_transform=False):
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

def stitch(input_folder, out_name):
	img_array = []
	for filename in glob.glob(f'{input_folder}/*.jpg'):
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)
	out = cv2.VideoWriter(f'{out_name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
	
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

#image_Size = 200
#image = np.array(cv2.imread('flower.jpg'))
#cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#proto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia")), cv2.COLOR_BGR2LAB)
#deuto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Deuteranope")), cv2.COLOR_BGR2LAB)
#tritano = cv2.cvtColor(np.uint8(CVD_Stim(image, "Tritanopia")), cv2.COLOR_BGR2LAB)
#cv2.imwrite('flower_protonopia.jpg', proto)
#cv2.imwrite('flower_deuteranope.jpg', deuto)
#cv2.imwrite('flower_tritanopia.jpg', tritano)

def vidwrite(fn, images, framerate=60, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

if __name__ == "__main__":
	image = np.array(cv2.imread('flower.jpg'))
	cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	proto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia")), cv2.COLOR_BGR2LAB)
	deuto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Deuteranope")), cv2.COLOR_BGR2LAB)
	tritano = cv2.cvtColor(np.uint8(CVD_Stim(image, "Tritanopia")), cv2.COLOR_BGR2LAB)
	cv2.imwrite('flower_sim_protonopia.jpg', np.uint8(cv2.cvtColor(np.uint8(proto), cv2.COLOR_LAB2BGR)))
	cv2.imwrite('flower_sim_deuteranope.jpg', np.uint8(cv2.cvtColor(np.uint8(deuto), cv2.COLOR_LAB2BGR)))
	cv2.imwrite('flower_sim_tritanopia.jpg', np.uint8(cv2.cvtColor(np.uint8(tritano), cv2.COLOR_LAB2BGR)))
	proto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia"  , simple_linear_transform = True)), cv2.COLOR_BGR2LAB)
	deuto = cv2.cvtColor(np.uint8(CVD_Stim(image, "Deuteranope" , simple_linear_transform = True)), cv2.COLOR_BGR2LAB)
	tritano = cv2.cvtColor(np.uint8(CVD_Stim(image, "Tritanopia", simple_linear_transform = True)), cv2.COLOR_BGR2LAB)
	cv2.imwrite('flower_protonopia.jpg', np.uint8(cv2.cvtColor(np.uint8(proto), cv2.COLOR_LAB2BGR)))
	cv2.imwrite('flower_deuteranope.jpg', np.uint8(cv2.cvtColor(np.uint8(deuto), cv2.COLOR_LAB2BGR)))
	cv2.imwrite('flower_tritanopia.jpg', np.uint8(cv2.cvtColor(np.uint8(tritano), cv2.COLOR_LAB2BGR)))

	#Fixing video for cvd
	SAVE_PATH = str(os.path.join(Path.home(), "Downloads"))
	youtube_video = YouTube('https://www.youtube.com/watch?v=sL7_FTrY4aQ', on_progress_callback=on_progress).streams
	title = youtube_video[0].title.replace('/','').replace('-','').replace(':','').replace(';','')+'.mp4'
	youtube_video.filter(file_extension='mp4', resolution='1080p').first().download(output_path = SAVE_PATH, filename = title)
	vid_file = str(os.path.join(SAVE_PATH, f"{title}"))
	print(SAVE_PATH, vid_file, title)
	print("EXISTS", Path(vid_file).exists())
	while not Path(vid_file).exists():
		print("Waiting 20 seconds for download")
		time.sleep(20)
	
	print("Processing Video")
	vidcap = cv2.VideoCapture(vid_file)
	images = []
	success,image = vidcap.read()
	print(image.shape)
	l, h, _ = image.shape
	image_transformed = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia")), cv2.COLOR_BGR2LAB)
	images.append(image_transformed)
	frame = 0
	while success:
		success,image = vidcap.read()
		if success == False:
			break
		frame += 1
		image_transformed = cv2.cvtColor(np.uint8(CVD_Stim(image, "Protanopia")), cv2.COLOR_BGR2LAB)#vid_file[:-4].replace(" ", "_")+
		images.append(image_transformed)
	
	vidwrite(f'{vid_file}', images)
