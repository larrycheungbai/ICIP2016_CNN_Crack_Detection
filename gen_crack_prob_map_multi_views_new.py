# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import math as math
import matplotlib.pyplot as plt
from PIL import Image
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os
#if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
caffe.set_mode_gpu()

from skimage import data, io, filters, transform

Polyp_root = '../'
current_folder = ''
if os.path.isfile(current_folder + '2016_TITS_DNN__iter_200000.caffemodel'):
    print 'Polyp network found.'
else:
    print 'Need to Download pre-trained CaffeNet model...'
    exit()
model_def = current_folder + 'deploy_crack.prototxt'
model_weights = current_folder + '2016_TITS_DNN__iter_200000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          99, 99)  # image size is 99x99
image = caffe.io.load_image('/home/lei/Downloads/caffe-master/data/PolypDetection/patches/ShortVD_wp_14_V2/ShortVD_wp_14_V2_frame_100_v_16_L_0.jpg')

patient_dir = "/home/lei/Downloads/caffe-master/examples/previous_project/mirror_images/"
for file in os.listdir(patient_dir):
    if file.endswith(".jpg") or file.endswith(".JPG") :
        print(file)
	image = caffe.io.load_image(patient_dir+file)
	(im_height,  im_width, channels) = image.shape
	print im_width
	print im_height
        w = im_width
	h = im_height
	results_name = file[0:len(file)-4]
	final_results_name = results_name+'_DNN_result'+'.txt'
	fout = open(final_results_name,"w")
	for column in range(49,w-49):
		for row in range(49,h-49):
#			print column, row
			image_patch =image[row-49:row+50, column-49:column+50]
#			(im_height,  im_width, channels) = image_patch.shape
#			print im_width, im_height
			transformed_image = transformer.preprocess('data', image_patch)
			net.blobs['data'].data[...] = transformed_image
			output = net.forward()
			output_prob_1 = output['prob'][0]
			pred_label_1 = output_prob_1.argmax()
#			print 'first', output_prob_1[0], output_prob_1[1],pred_label_1	
   		        image_patch_2 = transform.rotate(image_patch,90)
	   	        transformed_image = transformer.preprocess('data', image_patch_2)
 		        net.blobs['data'].data[...] = transformed_image
  		        output = net.forward()
   	                output_prob_2 = output['prob'][0]  # the output probability vector for the first image in the batch
			pred_label_2 = output_prob_2.argmax()
#			print 'second', output_prob_2[0], output_prob_2[1],pred_label_2

    			image_patch_3 = transform.rotate(image_patch,180)
	   	        transformed_image = transformer.preprocess('data', image_patch_3)
 		        net.blobs['data'].data[...] = transformed_image
  		        output = net.forward()
   	                output_prob_3 = output['prob'][0]  # the output probability vector for the first image in the batch
			pred_label_3 = output_prob_3.argmax()
#			print 'third', output_prob_3[0], output_prob_3[1],pred_label_3

	   	   	image_patch_4 = np.flipud(image_patch)
	   	        transformed_image = transformer.preprocess('data', image_patch_4)
 		        net.blobs['data'].data[...] = transformed_image
  		        output = net.forward()
   	                output_prob_4 = output['prob'][0]  # the output probability vector for the first image in the batch
			pred_label_4 = output_prob_4.argmax()
#			print 'forth', output_prob_4[0], output_prob_4[1],pred_label_4
			
		   	image_patch_5 = np.fliplr(image_patch)
	   	        transformed_image = transformer.preprocess('data', image_patch_5)
 		        net.blobs['data'].data[...] = transformed_image
  		        output = net.forward()
   	                output_prob_5 = output['prob'][0]  # the output probability vector for the first image in the batch
			pred_label_5 = output_prob_5.argmax()
#			print 'fifth', output_prob_5[0], output_prob_5[1],pred_label_5		
			aggregated_prob = (output_prob_1[1] + output_prob_2[1] + output_prob_3[1] + output_prob_4[1] + output_prob_5[1]) / 5
#                       print 'aggregate prob', aggregated_prob
			fout.write(str(aggregated_prob)+",")
		fout.write("\n")
	

