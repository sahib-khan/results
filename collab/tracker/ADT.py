#!/usr/bin/python

import vot
import sys
import collections
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.ops import control_flow_ops
import time
import numpy as np
import scipy.io as sio
import cv2, glob, os, re
import tracker_util as tutil


handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)


params = tutil.get_params()

param_path = '/root/collab/tracker/ADNet_params.mat'
results = {}
results['fps'] = []
results['location'] = []
results['overlap'] = []

nodes = {}
epoch = 0
initial_params = sio.loadmat(param_path)


#%% get here
with tf.Graph().as_default():
#%% get convolutional feature
    nodes['full_training'] = tf.placeholder_with_default(0.0, [])
    nodes['cropped_img'] = tf.placeholder_with_default(tf.cast(np.zeros([1,112,112,3]),tf.float32), [None, 112,112,3])
    nodes['cropped'] = tf.placeholder_with_default(1.0, [])
    nodes['boxes'] = tf.placeholder_with_default(tf.cast(np.zeros([1,4]),tf.float32), [None, 4])
    nodes['boxes_ind'] = tf.placeholder_with_default(tf.cast(np.zeros([1]),tf.int32), [None])
    nodes['image'] = tf.placeholder_with_default(tf.cast(np.zeros([1,112,112,3]),tf.float32), [None, None,None,3])
    nodes['crop'] = tf.image.crop_and_resize(nodes['image'], nodes['boxes'],  nodes['boxes_ind'], [112, 112])
    img = tf.cond(tf.equal(nodes['cropped'] ,0.0),
                  lambda : nodes['cropped_img'],
                  lambda : nodes['crop'])
    nodes['conv_feat'] = tutil.model_conv(img, param_path)
    
#%% get action and confidence
    nodes['conv'] = tf.placeholder_with_default(tf.cast(np.zeros([1,3,3,512]),tf.float32), [None, 3,3,512])
    input_feature = tf.cond(tf.equal(nodes['full_training'] ,1.0),
                            lambda : nodes['conv_feat'],
                            lambda : nodes['conv'])
    nodes['act_label'] = tf.placeholder_with_default(tf.cast(np.zeros([1,11]),tf.float32), [None, 11])
    nodes['sco_label'] = tf.placeholder_with_default(tf.cast(np.zeros([1,2]),tf.float32), [None, 2])
    nodes['action_hist'] = tf.placeholder_with_default(tf.cast(np.zeros([1,110]),tf.float32), [None, 110])
    nodes['is_training'] = tf.placeholder_with_default(0.0, [])
    train = tf.equal(nodes['is_training'],1.0)
    
    nodes['action'], conf, fc2 = tutil.model_fc(input_feature, nodes['action_hist'], train, param_path)
    nodes['soft_conf'] = tf.nn.softmax(conf)
    
#%% compute loss and train model
    nodes['act_loss'] = tf.losses.softmax_cross_entropy(nodes['act_label'], nodes['action'])
    nodes['sco_loss'] = tf.losses.softmax_cross_entropy(nodes['sco_label'], conf)
    
    learning_rate = tf.placeholder(tf.float32)
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    fc_tensor = [initial_params['fc1w'],initial_params['fc1b'][0],
                 initial_params['fc2w'],initial_params['fc2b'][0],
                 np.squeeze(initial_params['fc3w'].astype(np.float32)),
                 np.squeeze(initial_params['fc3b'].astype(np.float32)),
                 np.squeeze(initial_params['fc4w'].astype(np.float32)),
                 np.squeeze(initial_params['fc4b'].astype(np.float32)),]
    def assign(variables, fc_tensor):
        for i in range(6,14):
            variables[i].assign(fc_tensor[i-6])
        return variables
    
    assign_fc = tf.placeholder_with_default(0.0, [])
    v_ = tf.cond(tf.equal(assign_fc,1.0), lambda : fc_tensor,
                                          lambda : variables[6:14])
    fc_variables = []
    for i in range(6,14):
        fc_variables.append(variables[i].assign(v_[i-6]))
    
    nodes['act'] = tf.placeholder(tf.float32)
    losses = tf.cond(tf.equal(nodes['act'],1.0), lambda :   nodes['act_loss']+0*nodes['sco_loss'],
                                                 lambda : 0*nodes['act_loss']+  nodes['sco_loss'])
#    
    opimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = opimizer.compute_gradients(losses, var_list = variables[6:])
    for i in range(len(gradients)):
        if i%2 == 0:
            gradients[i] = (gradients[i][0]*10,gradients[i][1])
        else:
            gradients[i] = (gradients[i][0]*20,gradients[i][1])
    gradients[4] = (gradients[4][0]*2, gradients[4][1])
    gradients[5] = (gradients[5][0]*2, gradients[5][1])
        
    update_ops =[opimizer.apply_gradients(gradients, global_step=global_step)]
    update_op = tf.group(*update_ops)
    nodes['act_train_op'] = control_flow_ops.with_dependencies([update_op], losses, name='train_op')
    
    variables  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    reset = tf.placeholder_with_default(0.0, [])
    def do_nothing(variables):
        #do_nothing
        return variables
    def reset_momentum(variables):
        variables[15] = tf.assign(variables[15], tf.zeros_like(variables[15]))
        variables[16] = tf.assign(variables[16], tf.cast(0.9,dtype=tf.float32))
        variables[17] = tf.assign(variables[17], tf.cast(0.999,dtype=tf.float32))
        for i, v in enumerate(variables[18:]):
            variables[18+i] = tf.assign(v, tf.zeros_like(v))
        return variables
    variables = tf.cond(tf.equal(reset,1.0), lambda : reset_momentum(variables),
                                             lambda : do_nothing(variables))
    nodes['reset'] = reset
    nodes['learning_rate'] = learning_rate
#%% restore params and hardware setting
#    checkpoint_path = latest_checkpoint(train_dir)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # with tf.Session(config=config) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(fc_variables, feed_dict = {assign_fc : 1.0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(fc_variables, feed_dict = {assign_fc : 1.0})






class ADTracker(object):

	def __init__(self,img,gt):
		# l = i+1
		params['height'], params['width'] = img.shape[:2]
		self.f=0
		self.total_pos_data = {}
		self.total_neg_data = {}
		self.total_pos_action_labels = {}
		self.total_pos_examples = {}
		self.total_neg_examples = {}

		self.frame_window = []

		self.cont_negatives = 0

		self.full_history = []
		self.full_gt = []
		self.action_history_oh = np.zeros([params['num_action_history'],params['num_actions']]);
		                
		self.pos_examples = tutil.gen_samples('gaussian', gt, params['pos_init']*2,
		                                 params, params['finetune_trans'], params['finetune_scale_factor'])
		self.r = tutil.overlap_ratio(self.pos_examples,gt)
		self.pos_examples = self.pos_examples[np.where(self.r>params['pos_thr_init'])]
		self.pos_examples = self.pos_examples[np.random.choice(self.pos_examples.shape[0],
		                                             min(params['pos_init'], self.pos_examples.shape[0]),replace=False)]

		self.neg_examples = np.vstack((tutil.gen_samples('uniform', gt, params['neg_init'] , params, 1, 10),
		                         tutil.gen_samples('whole', gt, params['neg_init'] , params, 1, 10)))
		self.r = tutil.overlap_ratio(self.neg_examples,gt)
		self.neg_examples = self.neg_examples[np.where(self.r<params['neg_thr_init'])]
		self.neg_examples = self.neg_examples[np.random.choice(self.neg_examples.shape[0],
		                                             min(params['neg_init'], self.neg_examples.shape[0]),replace=False)]
		self.examples = np.vstack((self.pos_examples, self.neg_examples))
		self.feat_conv = tutil.get_conv_feature(sess, nodes['conv_feat'], 
		                                   feed_dict = {nodes['cropped'] : 1.0, nodes['boxes_ind'] : np.array([0]*self.examples.shape[0]),
		                                                nodes['image'] : [img], nodes['boxes'] : tutil.refine_box(self.examples, params)})
		self.pos_data = self.feat_conv[:self.pos_examples.shape[0]]
		self.neg_data = self.feat_conv[self.pos_examples.shape[0]:]

		self.pos_action_labels = tutil.gen_action_labels(params, self.pos_examples, gt)

		_ = sess.run(variables, feed_dict = {reset : 1.0})
		tutil.train_fc(sess, nodes, self.feat_conv, self.pos_action_labels,
		               params['iter_init'], params, params['init_learning_rate'])

		self.total_pos_data['%d'%self.f] = self.pos_data
		self.total_neg_data['%d'%self.f] = self.neg_data
		self.total_pos_action_labels['%d'%self.f] = self.pos_action_labels
		self.total_pos_examples['%d'%self.f] = self.pos_examples
		self.total_neg_examples['%d'%self.f] = self.neg_examples

		self.frame_window.append(self.f)
		self.is_negative = False

		self.action_history = np.zeros([params['num_show_actions']]);
		self.this_actions = np.zeros([params['num_show_actions'],11]);

		self.curr_bbox = gt.astype(np.float32)

		self.t = time.time()
		self.total_moves = 0
		self.move_counter = 0
		self.f+=1
        # check = 0

	def track(self,img):
		# print('entry')
		params['height'], params['width'] = img.shape[:2]
		curr_bbox_old = self.curr_bbox
		self.move_counter = 0
		target_score = 0

		num_action_step_max = 20;
		bb_step = np.zeros([num_action_step_max, 4])
		score_step = np.zeros([num_action_step_max, 1])
		self.is_negative = False
		prev_score = -9999
		self.this_actions = np.zeros([params['num_show_actions'],1])
		action_history_oh_old = self.action_history_oh

		while (self.move_counter < num_action_step_max):
		    bb_step[self.move_counter] = self.curr_bbox
		    score_step[self.move_counter] = prev_score;
		    
		    self.action_history_oh *= 0

		    for i, act in enumerate(self.action_history[:params['num_action_history']]):
		        if act<11:
		            self.action_history_oh[i,int(act)] = 1
		    
		    pred, pred_score = sess.run([nodes['action'], nodes['soft_conf']],
		                                feed_dict = {nodes['image'] : [img], nodes['cropped'] : 1.0,
		                                             nodes['full_training'] : 1.0, nodes['boxes_ind'] : np.array([0]), 
		                                             nodes['boxes'] : tutil.refine_box(np.expand_dims(self.curr_bbox,0), params),
		                                             nodes['action_hist'] : self.action_history_oh.reshape(1,-1)})
		    curr_score = pred_score[0,1]
		    max_action = np.argmax(pred[0])
		    if (curr_score < params['failedThre']):
		        self.is_negative = True;
		        curr_score = prev_score;
		        self.action_history[1:] = self.action_history[:-1]
		        self.action_history[0] = 12;
		        self.cont_negatives += 1;
		        break;
		        
		    self.curr_bbox = tutil.do_action(self.curr_bbox, max_action, params);
		    
		    if ((len(np.where(np.sum(np.equal(np.round(bb_step),np.round(self.curr_bbox)),1)==4)[0]) > 0)
		       & (max_action != params['stop_action'])):
		        max_action = params['stop_action']
		    
		    
		    self.action_history[1:] = self.action_history[:-1]
		    self.action_history[0] = max_action;
		    target_score = curr_score;
		    
		    if max_action == params['stop_action']:        
		        break
		    
		    self.move_counter += 1;
		    prev_score = curr_score;
		    
		#%% Tracking Fail --> Re-detection  
		if ( (self.f > 0) & (self.is_negative == True)):
		#                        print (f)
		#                        cv2.waitKey(0)
		    self.total_pos_data['%d'%self.f] = np.zeros([0,3,3,512])
		    self.total_neg_data['%d'%self.f] = np.zeros([0,3,3,512])
		    self.total_pos_action_labels['%d'%self.f] = np.zeros([0,11])
		    self.total_pos_examples['%d'%self.f] = np.zeros([0,4])
		    self.total_neg_examples['%d'%self.f] = np.zeros([0,4])
		    
		    samples_redet = tutil.gen_samples('gaussian', curr_bbox_old, params['redet_samples'], params, min(1.5, 0.6*1.15**self.cont_negatives), params['finetune_scale_factor'])
		    red_score_pred = sess.run(nodes['soft_conf'],
		                          feed_dict = {nodes['image']: [img], nodes['cropped'] : 1.0,
		                                       nodes['full_training'] : 1.0, nodes['boxes_ind'] : np.array([0]*samples_redet.shape[0]),
		                                       nodes['boxes'] : tutil.refine_box(samples_redet, params),
		                                       nodes['action_hist'] : np.vstack([self.action_history_oh.reshape(1,-1)]*samples_redet.shape[0]),
		                                       nodes['is_training'] : 0.0})


		    idx = np.lexsort((np.array(range(params['redet_samples'])),red_score_pred[:,1]))
		    target_score = np.mean(red_score_pred[(idx[-5:]),1])
		    if target_score > curr_score:
		        self.curr_bbox = np.mean(samples_redet[(idx[-5:]),:],0)
		    self.move_counter += params['redet_samples']

		#%% Tracking Success --> generate samples
		if ( (self.f > 0) & ((self.is_negative == False) | (target_score > params['successThre']))):
		    self.cont_negatives = 0;
		    self.pos_examples = tutil.gen_samples('gaussian', self.curr_bbox, params['pos_on']*2, params, params['finetune_trans'], params['finetune_scale_factor'])
		    self.r = tutil.overlap_ratio(self.pos_examples,self.curr_bbox)
		    self.pos_examples = self.pos_examples[np.where(self.r>params['pos_thr_on'])]
		    self.pos_examples = self.pos_examples[np.random.choice(self.pos_examples.shape[0],
		                                             min(params['pos_on'], self.pos_examples.shape[0]),replace=False)]
		    
		    self.neg_examples = tutil.gen_samples('uniform', self.curr_bbox, params['neg_on']*2, params, 2, 5)
		    self.r = tutil.overlap_ratio(self.neg_examples,self.curr_bbox)
		    self.neg_examples = self.neg_examples[np.where(self.r<params['neg_thr_on'])]
		    self.neg_examples = self.neg_examples[np.random.choice(self.neg_examples.shape[0],
		                                                 min(params['neg_on'], self.neg_examples.shape[0]),replace=False)]
		    
		    self.examples = np.vstack((self.pos_examples, self.neg_examples))
		    self.feat_conv = tutil.get_conv_feature(sess, nodes['conv_feat'],
		                                       feed_dict = {nodes['cropped'] : 1.0, nodes['boxes_ind'] : np.array([0]*self.examples.shape[0]),
		                                                    nodes['image'] : [img], nodes['boxes'] : tutil.refine_box(self.examples, params)})

		    self.total_pos_data['%d'%self.f] = self.feat_conv[:self.pos_examples.shape[0]]
		    self.total_neg_data['%d'%self.f] = self.feat_conv[self.pos_examples.shape[0]:]
		    
		    self.pos_action_labels = tutil.gen_action_labels(params, self.pos_examples, self.curr_bbox)
		    
		    self.total_pos_action_labels['%d'%self.f] = self.pos_action_labels
		    self.total_pos_examples['%d'%self.f] = self.pos_examples
		    self.total_neg_examples['%d'%self.f] = self.neg_examples

		    self.frame_window.append(self.f)
		    
		    if (len(self.frame_window) > params['frame_long']):
		        self.total_pos_data['%d'%self.frame_window[-params['frame_long']]] = np.zeros([0,3,3,512])
		        self.total_pos_action_labels['%d'%self.frame_window[-params['frame_long']]] = np.zeros([0,11])
		        self.total_pos_examples['%d'%self.frame_window[-params['frame_long']]] = np.zeros([0,4])
		        
		    if (len(self.frame_window) > params['frame_short']):
		        self.total_neg_data['%d'%self.frame_window[-params['frame_short']]] = np.zeros([0,3,3,512])
		        self.total_neg_examples['%d'%self.frame_window[-params['frame_short']]] = np.zeros([0,4])
		        
		#%% Do online-training
		if ( ((self.f+1)%params['iterval'] == 0)  | (self.is_negative == True) ):
		    if (self.f+1)%params['iterval'] == 0:
		        f_st = max(0,len(self.frame_window)-params['frame_long'])
		        
		        self.pos_data = []
		        self.pos_action_labels = []
		        for wind in self.frame_window[f_st:]:
		            self.pos_data.append(self.total_pos_data['%d'%wind])
		            self.pos_action_labels.append(self.total_pos_action_labels['%d'%wind])
		            
		        self.pos_data = np.vstack(self.pos_data)
		        self.pos_action_labels = np.vstack(self.pos_action_labels)
		        
		    else:
		        f_st = max(0,len(self.frame_window)-params['frame_short'])
		        
		        self.pos_data = []
		        self.pos_action_labels = []
		        for wind in self.frame_window[f_st:]:
		            self.pos_data.append(self.total_pos_data['%d'%wind])
		            self.pos_action_labels.append(self.total_pos_action_labels['%d'%wind])
		            
		        self.pos_data = np.vstack(self.pos_data)
		        self.pos_action_labels = np.vstack(self.pos_action_labels)
		    
		    f_st = max(0,len(self.frame_window)-params['frame_short'])
		    self.neg_data = []
		    for wind in self.frame_window[f_st:]:
		        self.neg_data.append(self.total_neg_data['%d'%wind])
		        
		    self.neg_data = np.vstack(self.neg_data)
		    
		    self.feat_conv = np.vstack((self.pos_data, self.neg_data))
		#                            if check == 5:
		    _ = sess.run(variables, feed_dict = {reset : 1.0})
		#                                check = 0
		    iteration = params['iter_on']
		#                            if self.is_negative:
		#                                iteration = params['iter_on']//2
		    tutil.train_fc(sess, nodes, self.feat_conv, self.pos_action_labels,
		                   iteration, params, params['on_learning_rate'])
		    
		self.full_history.append(self.curr_bbox)
		self.full_gt.append(gt)
		self.total_moves += self.move_counter

		frame = np.copy(img)
		# frame = cv2.rectangle(frame,(int(gt[0]),int(gt[1])),
		#                             (int(gt[0]+gt[2]),int(gt[1]+gt[3])),[0,0,255],2)
		frame = cv2.rectangle(frame,(int(self.curr_bbox[0]),int(self.curr_bbox[1])),
		                            (int(self.curr_bbox[0]+self.curr_bbox[2]),int(self.curr_bbox[1]+self.curr_bbox[3])),[255,0,0],2)

		# cv2.imwrite('results/'+frames[self.f][-8:],frame)
		# cv2.imshow('f',frame)
		# key = cv2.waitKey(1) & 0xff
		# if key == ord('s'):
		# 	return
		self.f+=1
		max_val=.99
		return vot.Rectangle(self.curr_bbox[0], self.curr_bbox[1], self.curr_bbox[2], self.curr_bbox[3]), max_val
		# print('reached')
    









image = cv2.imread(imagefile)
gt1=[int(selection.x),int(selection.y),int(selection.width),int(selection.height)]
gt = np.array(gt1,dtype=int)
tracker = ADTracker(image, gt)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region, confidence = tracker.track(image)
    handle.report(region,confidence)


sess.close()
cv2.destroyAllWindows()
