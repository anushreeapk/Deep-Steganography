import os
from shutil import copyfile


f = open('GOP.format','r')
gop = f.readlines()
p = 0
i = 0
b = 0
frame_no = 0
if not os.path.exists('combined_frames'):
        os.makedirs('combined_frames')
for line in gop:

        frame_type = line.split('=')[1]
        #print(frame_type.strip())
        dir_name = frame_type.strip('\n')+"_frames_compress/"
        #print(dir_name)
        #print(frame_type.strip())
        if frame_type.strip('\n') =="B":
                b+=1
                filepath = os.path.join(dir_name,"frame"+'{0:03}'.format(b)+".jpg")
                print(filepath)
        if frame_type.strip('\n') =="I":
                i+=1
                filepath = os.path.join(dir_name,"frame"+'{0:03}'.format(i)+".jpg")
                print(filepath)
        if frame_type.strip('\n') =="P":
                p+=1
                filepath = os.path.join(dir_name,"frame"+'{0:03}'.format(p)+".jpg")
                print(filepath)
        if os.path.exists(filepath):
                frame_no+=1
                dst = os.path.join('combined_frames/','frame'+'{0:03}'.format(frame_no)+".jpg")
                copyfile(filepath, dst)

