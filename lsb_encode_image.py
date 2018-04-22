import os
from glob import glob
from os.path import expanduser, join
from stegano import lsb
import scipy.misc
import cv2
import numpy as np

class Prep_Network():

	def __init__(self):
		self.secret_image_data_list = []
		self.designated_size = (160, 120)

	def encode_text(self, video_dir):

		self.GOP_file = glob(join(video_dir, '*.txt'))[0]
		self.video_dir = video_dir
		self.video_name = self.video_dir.split("/")[-1]
		self.p_frames_dir = os.path.join(video_dir, self.video_name+".P.frames")
		self.b_frames_dir = os.path.join(video_dir, self.video_name+".B.frames")
		self.i_frames_dir = os.path.join(video_dir, self.video_name+".I.frames")
		print(self.GOP_file, self.p_frames_dir, self.i_frames_dir, self.b_frames_dir)

		p_frame_count = 0
		i_frame_count = 0
		b_frame_count = 0
		total_count = 0

		self.secret_image_data_list = []

		with open(self.GOP_file, "r") as fp:
			for line in fp:
				total_count +=1
				line = line.strip()
				if line == "P":
					p_frame_count+=1
					text_to_be_encoded = "P " + str(total_count)
					image_to_be_encoded = os.path.join(self.p_frames_dir, "frame%03d"%p_frame_count+".jpg")
				if line == "I":
					i_frame_count+=1
					text_to_be_encoded = "I " + str(total_count)
					image_to_be_encoded = os.path.join(self.i_frames_dir, "frame%03d"%i_frame_count+".jpg")
				if line == "B":
					b_frame_count+=1
					text_to_be_encoded = "B " + str(total_count)
					image_to_be_encoded = os.path.join(self.b_frames_dir, "frame%03d"%b_frame_count+".jpg")
				
				secret_image = lsb.hide(image_to_be_encoded, text_to_be_encoded) #, generators.eratosthenes())
				
				secret_image_data = np.array(secret_image)
				if secret_image.size != self.designated_size:
					cv2.resize(secret_image_data, self.designated_size)
				self.secret_image_data_list.append(secret_image_data)

				print(total_count)

		self.secret_image_data_list = np.array(self.secret_image_data_list)

	def stitch_images(self, image_counter):

		width_number_of_images = int(np.floor(np.sqrt(len(self.secret_image_data_list))))
		height_number_of_images = np.ceil(len(self.secret_image_data_list)/width_number_of_images)
		stacked_counter = 0
		stitched_secret_image = np.array([])
		count = 0
		while stacked_counter <= len(self.secret_image_data_list):
			count+=1
			print(count)
			row_stacked_image = np.hstack(self.secret_image_data_list[stacked_counter:stacked_counter+width_number_of_images])
			if count == height_number_of_images:
				print(type(row_stacked_image.shape[1]), row_stacked_image.shape[1]/self.designated_size[0])
				blank_image_filling = np.full((self.designated_size[1],self.designated_size[0]*(width_number_of_images - int(row_stacked_image.shape[1]/self.designated_size[0])),3),255)
				row_stacked_image = np.hstack([row_stacked_image, blank_image_filling])

			if not len(stitched_secret_image):
				stitched_secret_image = row_stacked_image
			else:
				stitched_secret_image = np.vstack([stitched_secret_image, row_stacked_image])
			stacked_counter+=width_number_of_images
		scipy.misc.imsave("secret_"+str(image_counter)+".png",stitched_secret_image)


if __name__ == "__main__":
	pp = Prep_Network()

	videos = glob(join("frames_dataset","*"))
	i=0
	for video_dir in videos:
		print(video_dir)
		pp.encode_text(video_dir)
		print ("Encoding Done!")
		pp.stitch_images(i)
		i+=1
