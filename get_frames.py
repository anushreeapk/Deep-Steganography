import cv2
import os
#ffmpeg -i <inputfile> -vf '[in]select=eq(pict_type\,B)[out]' b.frames.mp4
import subprocess
import pdb


def getGOP(filename,name):
  print(filename)
  #ffprobe -show_frames "output.mp4" | grep pict_type > GOP.format
  result = subprocess.Popen(["ffprobe","-show_frames",filename],
  stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  #pdb.set_trace()
  gop_path = os.path.join('frames_dataset/'+name,name+'_GOP.txt')
  f = open(gop_path,'w')
  for x in result.stdout.readlines():
         if b'pict_type' in x:
              #pdb.set_trace()
              x = str(x).split('=')
              seq = x[1].strip('\\n\'')
              f.write(str(seq)+'\n')

def getLength(filename):
  print(filename)
  #ffmpeg -i file.flv 2>&1 | grep "Duration"
  result = subprocess.Popen(["ffmpeg","-i",filename,"2>&1","|","grep","Duration"],
  stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  #pdb.set_trace()
  return [x for x in result.stdout.readlines() if b'Duration' in x]

def compress_video(filename):
	name = filename.strip('.avi')
	filepath = os.path.join('mini_dataset/',filename)
	compress_path = os.path.join('mini_dataset/',name+'_comp.mp4')
	print("Compressing file.....")
	subprocess.call(['ffmpeg', '-i',filepath, '-vcodec', 'libx264','-crf','20',compress_path])


def get_frames(ftype,filename):
	name = filename.strip('.avi')
		
	filepath = os.path.join('mini_dataset/',filename)
	compress_path = os.path.join('mini_dataset/',name+'_comp.mp4')
	#Compress the Video(Vary CRF between 18-24)
	#ffmpeg -i person14_jogging_d3_uncomp.avi -vcodec libx264 -crf 20 output.mp4
	#pdb.set_trace()
	if not os.path.exists('frames_dataset'):
		os.makedirs('frames_dataset')
	if not os.path.exists('frames_dataset/'+name):
		os.makedirs('frames_dataset/'+name)
	#Get Video File duration
	duration = getLength(filepath)
	dur = str(duration).split()
	dur = dur[2].strip(',')
	frames_dest = os.path.join('frames_dataset/'+name,name+'.'+ftype+'.frames')
	if not os.path.exists(frames_dest):
		os.makedirs(frames_dest)
	
	#Get P,I,B frames
	#ffmpeg -ss 0 -i person14_jogging_d3_uncomp.avi -t 24 -q:v 2 -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0 I_frames/frame%03d.jpg
	subprocess.call(['ffmpeg', '-ss','0','-i',compress_path, '-t',dur,'-q:v','2','-vf','select=eq(pict_type\,PICT_TYPE_'+ftype.upper()+')','-vsync','0',frames_dest+'/frame%03d.jpg'])
	
	getGOP(compress_path,name)

folder = 'mini_dataset'
for the_file in os.listdir(folder):
	compress_video(the_file)
	filepath = os.path.join(folder,the_file)
	print(filepath)
	get_frames('B',the_file)
	get_frames('I',the_file)
	get_frames('P',the_file)
	print("******************************************")
print("Done!")
