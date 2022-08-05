"""
original repository:
https://github.com/Rudrabha/Wav2Lip

additionally needed:
https://github.com/1adrianb/face-alignment

res10_300x300_ssd_iter_140000.caffemodel
deploy.prototxt.txt

imutils

# --- additional parameters: ---#
--align_face / default off
--preview / default off
--no_rotation / default on
--loop / default off
--pingpong / default off
--no_caffe / default off
--img_sequence / default off - writes result as image sequence / output to ./seq
--face_sequence / default off - read numbered image-sequence / filename should be first image name (../000.png)

"""


from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import face_detection
from models import Wav2Lip
import platform

#- added
import imutils
import face_alignment
import shutil

#from torch.cuda.amp import autocast


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
					
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, help='If True, then use only first video frame for inference', default=False)

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 29.97)', default=29.97, required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least') # default=[0, 10, 0, 0] = caffe [-15 5 0 0]

# no more functional / set to 1
parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)

# no more functional / set to 1					
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=1) #128

parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.''Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=True, action='store_true',help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.''Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--align_face', default=False, action='store_true',help='Align face for better inference results')

parser.add_argument('--loop', default=False, action='store_true',help='Loop video on longer audio')

parser.add_argument('--preview', default=False, action='store_true',help='Preview during inference')

parser.add_argument('--no_rotation', default=False, action='store_true',help='Shows preview frame to horizontically align the face in the video')

parser.add_argument('--pingpong', default=False, action='store_true',help='Loop video backwards/forward on longer audio')

parser.add_argument('--no_caffe', default=False, action='store_true',help='don not use caffemodel for face detection')

parser.add_argument('--img_sequence', default=False, action='store_true',help='Save result as image sequence in png format')

parser.add_argument('--face_sequence', default=False, action='store_true',help='use image sqeuence not video that contains the face to use')



args = parser.parse_args()
args.img_size = 96
args.wav2lip_batch_size = 1
mel_step_size = 16
btn_down = False
pre_angle = 0
loop = False

#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

net = cv2.dnn.readNetFromCaffe('caffemodel/deploy.prototxt.txt', 'caffemodel/res10_300x300_ssd_iter_140000.caffemodel')

alignment = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for face alignment.'.format(device))
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device = alignment, flip_input=True, face_detector='sfd') #device='cpu/cuda'

	
if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True
	
if args.face_sequence:
	args.static = False

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			#raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		face = image[y1: y2, x1:x2]
		#cv2.imshow("DetFace",face)
		#cv2.waitKey()
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	#cv2.destroyAllWindows()
	return results
	

def face_detect_caffe(images):
	predictions = []
	for i in tqdm(range(0,len(images))):
			
		im = np.array(images[i])
		(h,w) = im.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		
		rect = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
				
		(x1,y1,x2,y2) = rect.astype('int')
				
		predictions.append((x1,y1,x2,y2))

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	
	os.system('cls')
	print ("Detection done")
	frame_count = 0
	fc = 0
	for rect, image in zip(predictions, images):
		if fc == 0:
			face = image[y1: y2, x1:x2]
			#cv2.imwrite('temp/first_face.jpg', face)
			fc = 1
		if rect is None:
			print ("No face detected in Frame number: " + str(frame_count))
			x1=0
			y1=0
			x2=w #96
			y2=h #96		

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		# - always crop square image
		
		breite = x2 - x1
		hohe = y2 - y1
		
		#diff = (hohe - breite)  # - crop square
		diff = (hohe - breite)//2  # - crop nearly square
		
		#y1 = max((0, rect[1] - pady1)) # - no square
		y1 = max((0, rect[1] - pady1) + (diff))
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
				
		# - if Rect > Frame
		
		if  x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h or breite <1 or hohe < 1:
			print ("Detection error in Frame number: " + str(frame_count))
			x1=0
			y1=0
			x2=w #96
			y2=h #96
		
		frame_count += 1
		
				
		face = image[y1: y2, x1:x2]
		#cv2.imshow("DetFace",face)
		#cv2.waitKey(10)
		 						
		results.append([x1, y1, x2, y2])		
		
	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
	#cv2.destroyAllWindows()
	return results


def correct_angle(input_img):
	global angle
	global autoalign
    
	try:   
		preds = fa.get_landmarks(input_img)[-1]
	except:
		autoalign = False
		print ("Alignment warning")
		angle = 0
		return angle
        
	lEyeY = (preds[36, 0])
	lEyeX = (preds[36, 1])
	rEyeY = (preds[45, 0])
	rEyeX= (preds[45, 1])

	dY=(rEyeY-lEyeY)
	dX=(rEyeX-lEyeX)

	new_angle = int(np.degrees(np.arctan2(dY, dX)) - 90)
		 
	return new_angle



def datagen(frames, mels):
	os.system('cls')
	print("Detecting face...")
	global angle
	global ori_face
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		
		# -caffe or not
		if not args.no_caffe:
			if not args.static:
				face_det_results = face_detect_caffe(frames)
			else:
				face_det_results = face_detect_caffe([frames[0]])
		else:
			if not args.static:
				face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
			else:
				face_det_results = face_detect([frames[0]])
				
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
		
	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()
		
		try:
			face = cv2.resize(face, (args.img_size, args.img_size)) # 96x96
		except:
			print ("Error, trying to resolve...")
			continue
		
		## - alignment rotate
		
		if args.align_face:
			angle = correct_angle(face)
			M = cv2.getRotationMatrix2D((48,32), angle*-1, 1.0) # w/h rotation point 1/3 height
			face = cv2.warpAffine(face, M, (args.img_size,args.img_size))
			## or
			##face = imutils.rotate(face, angle *-1) # rotate detected face
			#cv2.imshow("Face",face)
			#cv2.waitKey(1)
		
		## -	
			 
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
		yield img_batch, mel_batch, frame_batch, coords_batch
			

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint


def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)
	model = model.to(device)
	#model = model.half() 
	return model.eval()


## - Draw line on source video

def get_points(im):

  # Set up data to send to mouse handler
  data = {}
  data['im'] = im.copy()
  data['lines'] = []

  # Set the callback function for any mouse event
    
  cv2.setMouseCallback("unique_window_identifier", mouse_handler, data)
  cv2.imshow("unique_window_identifier",im)
  cv2.setWindowTitle("unique_window_identifier", "Draw left to right eye(clockwise) - any key to accept")
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # Convert array to np.array in shape n,2,2
  points = np.uint16(data['lines'])

  return points, data['im']

def mouse_handler(event, x, y, flags, data):
	global pre_angle,ix,iy,ex,ey,btn_down

	if event == cv2.EVENT_LBUTTONUP and btn_down:
	#if you release the button, finish the line
		btn_down = False
		data['lines'][0].append((x, y)) #append the seconf point
		cv2.circle(data['im'], (x, y), 7, (0, 0, 255),1)
		cv2.line(data['im'], data['lines'][0][0], data['lines'][0][1], (0,0,255), 1)
		cv2.imshow("unique_window_identifier", data['im'])
		ex = x
		ey = y
		dX=(ey-iy)
		dY=(ex-ix)
		pre_angle = int(np.degrees(np.arctan2(dY, dX)) - 90)
		print (pre_angle)
		#cv2.destroyAllWindows()
	elif event == cv2.EVENT_MOUSEMOVE and btn_down:
		#thi is just for a ine visualization
		image = data['im'].copy()
		cv2.line(image, data['lines'][0][0], (x, y), (0,0,255), 1)
		cv2.imshow("unique_window_identifier", image)

	elif event == cv2.EVENT_LBUTTONDOWN and len(data['lines']) < 9:   #anzahl versuche
		btn_down = True
		data['lines'].insert(0,[(x, y)]) #prepend the point
		cv2.circle(data['im'], (x, y), 7, (0, 0, 255), 1)
		cv2.imshow("unique_window_identifier", data['im'])
		cv2.setWindowTitle("unique_window_identifier", "Draw left to right eye(clockwise) - any key to accept")
		ix = x
		iy = y
		
		## -

def main():
	global ori_face
	global angle
	global pre_angle
	global loop
	angle = 0
	pre_angle = 0
	#end_mask = cv2.imread('white.jpg')
	end_mask = 255 * np.ones((96, 96, 3), dtype=np.uint8)
	#cv2.imshow("M",end_mask)
	#cv2.waitKey()

	#- different masks
	#src_mask = np.zeros((96, 96, 3), dtype=np.uint8)
	#src_mask = cv2.ellipse(src_mask, (48,72), (30,20),0,0,360,(255,255,255), -1)
	#src_mask = cv2.GaussianBlur(src_mask,(5,5),cv2.BORDER_DEFAULT)
	#src_mask = cv2.cvtColor(src_mask, cv2.COLOR_RGB2GRAY)
	#src_mask = np.reshape(src_mask, [src_mask.shape[0],src_mask.shape[1],1])
	#src_mask = src_mask /255

	#- mouth
	src_mask = np.zeros((192,192), dtype=np.uint8) #96,96
	#src_mask = cv2.rectangle(src_mask,(2,2),(190,190),(255,255,255), -1)
	#src_mask = cv2.ellipse(src_mask, (96,144), (60,40),0,0,360,(255,255,255), -1)
	#src_mask = cv2.ellipse(src_mask, (96,144), (60,43),0,0,360,(255,255,255), -1)
	
	#src_mask = cv2.ellipse(src_mask, (96,146), (60,45),0,0,360,(255,255,255), -1) #60,45#
	src_mask = cv2.ellipse(src_mask, (96,136), (75,55),0,0,360,(255,255,255), -1)
	src_mask = cv2.cvtColor(src_mask, cv2.COLOR_GRAY2RGB)/255
	
	#- bbox
	#bbox_mask = np.zeros((192,192), dtype=np.uint8) #96,96
	#bbox_mask = cv2.rectangle(bbox_mask,(2,2),(188,188),(255,255,255), -1)
	#bbox_mask = cv2.cvtColor(bbox_mask, cv2.COLOR_GRAY2RGB)/255			
	
	#cv2.imshow("M",src_mask)
	#cv2.waitKey()
		
	#src_mask = cv2.resize(src_mask, (args.img_size, args.img_size))
	cx = args.img_size//2
	cy = args.img_size//2
	
	if  os.path.exists('seq'): # png jpg bmp
		shutil.rmtree('seq')	
	if args.img_sequence:
		if not os.path.exists('seq'):
			os.mkdir('seq')
	
					
	if not os.path.isfile(args.face) and not args.face_sequence:
		raise ValueError('--face argument must be a valid path to video/image file')
		
	elif not args.face_sequence and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	#if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:

		full_frame = cv2.imread(args.face)
		full_frames = [full_frame]
		fps = args.fps
		vh, vw = full_frames[0].shape[:-1]
		#src_mask = cv2.imread('maskesoft.jpg')
		#src_mask = cv2.resize(src_mask, (args.img_size, args.img_size))
		
		## - draw angle - image
		
		if not args.no_rotation:
			#still_reading, frame = video_stream.read()
			cv2.imshow("unique_window_identifier", full_frame)
			pts, final_image = get_points(full_frame)
			cv2.destroyAllWindows()
			full_frame = imutils.rotate_bound(full_frame, pre_angle)
			full_frames = [full_frame]
			
				
	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		
		if args.face_sequence:
			fps = args.fps
			#if args.fps == 0:
			#	fps = 25
			
		vw = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		vh = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))+1

		duration = frame_count/fps

			#duration = frame_count/fps
					
   ## - draw angle - video
    
		if not args.no_rotation:
			still_reading, frame = video_stream.read()
			cv2.imshow("unique_window_identifier", frame)
			pts, final_image = get_points(frame)
			cv2.destroyAllWindows()
		
		## -
   
		if args.outfile == 'temp/silence.mp4':
			command = 'ffmpeg -y -f lavfi -i anullsrc -t ' + str(duration) + ' -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 temp/silence.mp3'
			subprocess.call(command, shell=platform.system() != 'Windows')
			
		
		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			#if args.rotate:
			#	frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
				
			frame = imutils.rotate_bound(frame, pre_angle) #  args.angle drehen
			
			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]
			full_frames.append(frame)
			
	if args.pingpong:
		loop = True
		full_frames_rev = full_frames[::-1] #reverse
		full_frames = full_frames + full_frames_rev
	
	if args.loop:
		loop = True
		
	print ("Number of frames available for inference: "+str(len(full_frames)))
	
	n_frames = (len(full_frames))

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'
		
	wav = audio.load_wav(args.audio, 16000) # 16000
		
	mel = audio.melspectrogram(wav)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []

	mel_idx_multiplier = 80./fps
	
	i = 0
	
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)
	#pred_out = cv2.VideoWriter("temp/predictions.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (96, 96))
	framecount = 0
	
	# +++++
	#total=int(np.ceil(float(len(mel_chunks))/batch_size))
	#model = load_model(args.checkpoint_path)
	#print ("Model loaded, writing temp video...")
			
	#if args.preview:
	#	print ("Press 'Esc' to stop and save...")

	#frame_h, frame_w = full_frames[0].shape[:-1]
	#out = cv2.VideoWriter('temp/temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw, vh))
	# ++++++
	
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded, writing temp video...")
			
			if args.preview:
				print ("Press 'Esc' to stop and save...")

			frame_h, frame_w = full_frames[0].shape[:-1]

			
			out = cv2.VideoWriter('temp/temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw, vh))
			#out = cv2.VideoWriter('temp/temp.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (vw, vh))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			#with autocast():
			pred = model(mel_batch, img_batch)

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			#print (c)
			
			#pred_out.write(p.astype(np.uint8))				
			
			orig_face = f[y1:y2, x1:x2]
			#orig_face = cv2.resize(orig_face,(args.img_size, args.img_size))
			
			## - ######################Biggest Sh.t ever ###########################

			if args.align_face:
							
				src_maskb = src_mask.copy()
				src_maskb = cv2.resize(src_maskb,(args.img_size, args.img_size))
				M = cv2.getRotationMatrix2D((48,32), angle, 1.0) # w/h rotation point rotate back
				src_maskb = cv2.warpAffine(src_maskb, M, (args.img_size,args.img_size))
				p = cv2.warpAffine(p, M, (args.img_size,args.img_size))
				
				## or
				##src_maskb = imutils.rotate(src_maskb, angle)
				##p = imutils.rotate(p, angle)
				
				#cv2.imshow("sm",src_maskb)
				src_maskb = cv2.resize(src_maskb,(x2 - x1, y2 - y1))			
				orig_face = cv2.resize(orig_face,(args.img_size, args.img_size))
				
				gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
				ret,f_mask = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
				f_mask = f_mask.astype(np.uint8)
				kernel = np.ones((2, 2), 'uint8') #h, w
				f_mask = cv2.erode(f_mask, kernel, iterations=1)
				p = (cv2.seamlessClone(p, orig_face, f_mask, (cx,cy), cv2.NORMAL_CLONE))

				orig_face = f[y1:y2, x1:x2]
			
			src_mask = cv2.resize(src_mask,(x2 - x1, y2 - y1))
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1),interpolation=cv2.INTER_LANCZOS4)


			blurfactor = (x2 - x1)//80
			if blurfactor % 2 == 0:
				blurfactor = blurfactor + 1
			
			if not args.align_face:
				src_mask = cv2.GaussianBlur(src_mask,(blurfactor,blurfactor),cv2.BORDER_DEFAULT)
				p = p * src_mask + orig_face * (1 - src_mask)
			
			if args.align_face:
				src_maskb = cv2.GaussianBlur(src_maskb,(blurfactor,blurfactor),cv2.BORDER_DEFAULT)
				p = p * src_maskb + orig_face * (1 - src_maskb)			
				
				
			## - ############################### EOS ##########################################
			
			#p = p.astype(np.uint8)
			#end_mask = cv2.resize(end_mask.astype(np.uint8), (x2 - x1, y2 - y1))
			#centerX = x1+((x2-x1)//2)
			#centerY = y1+((y2-y1)//2)
			#cc = (centerX,centerY)
			#f = (cv2.seamlessClone(p, f, end_mask, cc, cv2.NORMAL_CLONE))
			## - ##############################################################################
			f[y1:y2, x1:x2] = p
			## - ##############################################################################
			
			f = imutils.rotate_bound(f, pre_angle *-1) # fullframe zuruckdrehen preangle!!

			#cv2.imshow("zuruck",f)
			#cv2.waitKey()
			
			(h,w) = f.shape[:2]
			
			xx1 = (w - vw)//2
			yy1 = (h -vh)//2
			xx2 = xx1 + vw
			yy2 = yy1 + vh
						
			f = f[yy1:yy2, xx1:xx2] # alte ausschnitt grosse widerherstellen
			out.write(f)
			
			if args.img_sequence:
				cv2.imwrite(os.path.join('seq', '{:0>7d}.png'.format(framecount)), f)
			
			framecount +=1
			if args.preview:
				cv2.imshow(args.outfile,f)
				k = cv2.waitKey(1)
				if k == 27:
					cv2.destroyAllWindows()
					out.release()
					command = 'ffmpeg -y -i ' + '"' + args.audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'
					subprocess.call(command, shell=platform.system() != 'Windows')
				
					sys.exit()
      
      ## - stop on video end or loop video if audio is longer duration
         			
			if n_frames > 1 and framecount > n_frames and not loop:
				if args.outfile == 'temp/silence.mp4':
					sys.exit()
				out.release()
				command = 'ffmpeg -y -i ' + '"' + args.audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'
				
				subprocess.call(command, shell=platform.system() != 'Windows')
				sys.exit()
				
	out.release()
	
	command = 'ffmpeg -y -i ' + '"' + args.audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'
	
	os.system('cls')
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
