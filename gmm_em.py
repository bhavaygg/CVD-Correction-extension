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
import subprocess

def stitch(input_folder, out_name):
	'''
	Function to combine a folder of images to video
	Args:
		- input_folder: Path to folder
		- out_name: Name of output video file
	'''
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

def vidwrite(fn, images, framerate=60, vcodec='libx264'):
	'''
	Function to combine an array of images to video
	Args:
		- fn: Name of output video file
		- images: array of images
		- framerate: fps of desired video
	'''
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
	video_link = 'https://www.youtube.com/watch?v=sL7_FTrY4aQ'
	SAVE_PATH = str(os.path.join(Path.home(), "Downloads"))
	youtube_video = YouTube(video_link, on_progress_callback=on_progress).streams
	title = youtube_video[0].title.replace('/','').replace('-','').replace(':','').replace(';','').replace(' ', '_')+'.mp4'
	youtube_video.filter(file_extension='mp4', resolution='1080p').first().download(output_path = SAVE_PATH, filename = title)
	vid_file = str(os.path.join(SAVE_PATH, f"{title}"))
	#print(SAVE_PATH, vid_file, title)
	#print("EXISTS", Path(vid_file).exists())
	while not Path(vid_file).exists():
		print("Waiting 20 seconds for download")
		time.sleep(20)
	print("Processing Video")
	try:
		os.makedirs(Path(vid_file[:-4]))
	except:
		pass
	vidcap = cv2.VideoCapture(vid_file)
	images = []
	frame = 0
	img_path = os.path.join(vid_file[:-4], "current_frame.jpg")
	success,image = vidcap.read()
	cv2.imwrite(img_path, np.uint8(cv2.cvtColor(np.uint8(image), cv2.COLOR_LAB2BGR)))
	subprocess.run(["matlab", "-nosplash", "-wait", "-r", f"demo {img_path} Protanopia {str(Path(vid_file[:-4]))} {frame:05d}; exit;", "-nodesktop", "-minimize", "&"])
	l, h, _ = image.shape
	while success:
		success,image = vidcap.read()
		if success == False:
			break
		frame += 1
		cv2.imwrite(img_path, np.uint8(cv2.cvtColor(np.uint8(image), cv2.COLOR_LAB2BGR)))
		subprocess.run(["matlab", "-nosplash", "-wait", "-r", f"demo {img_path} Protanopia {str(Path(vid_file[:-4]))} {frame:05d}; exit;", "-nodesktop", "-minimize", "&"])
		frame+=1
	os.remove(img_path)
	os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{str(Path(vid_file[:-4]))}\*.png' -c:v libx264 -pix_fmt yuv420p {vid_file}")
