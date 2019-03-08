import numpy as np
np.random.seed(seed=0) 

import time
import os
import pprint
pp = pprint.PrettyPrinter()

import random
import tensorflow as tf
import math
from copy import deepcopy
from pdb import set_trace as debug
from imp import load_source

import ROOT
ROOT.gROOT.SetBatch(True)

from helpers.tf_helpers import *
from helpers.train_config import trainingParameters
from helpers.config import GlobalNoiseThreshold, SignalSmearing, LogScale


#training command on pclcdgpu: python training.py --EpochStart 0  --Nepochs 150 --checkpoint_dir ~/G5_C1_E3_P2_test --input_dir /afs/cern.ch/work/t/tquast/public/Sept2017_HGCALTB_Sim
flags.DEFINE_integer("EpochStart", 0, "First epoch to start the training from. [0]")
flags.DEFINE_integer("Nepochs", 150, "Epochs to train [150]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [256]")
flags.DEFINE_integer("eval_sample_size", 2560, "The size of samples for evaluation images [2560]")
flags.DEFINE_integer("height", 12, "The height size of image to use. [12]")
flags.DEFINE_integer("width", 15, "The width size of image to use. [15]")
flags.DEFINE_integer("color_dim", 7, "The dimension of the colours - equivalent to the number of layers of the prototype. (7 EE layers + 10 FH) [7]")
flags.DEFINE_integer("z_dim", 10, "The number of entries for the z-vector which represents the noise. [10]")

flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory name to save the checkpoints [./checkpoint]")
flags.DEFINE_string("input_dir", "/eos/user/t/tquast/outputs/Testbeam/July2017/numpy", "Directory in which the input files are stored. [/eos/user/t/tquast/outputs/Testbeam/July2017/numpy]")
flags.DEFINE_string("log_dir", None, "Directory name to save the log file for tensorboard. If set to None, the checkpoint_dir is taken. [None]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("loadCounter", -1, "Counter to load for the evaluation. [-1]")
flags.DEFINE_float("PositionForEval", 20, "Energy for evaluation of pion showers [20./20.]")
flags.DEFINE_boolean("finalSave", False, "Save the output at the end? [False]")

flags.DEFINE_integer("GPUIndex", 0, "Index of the GPU used for training. [0]")

FLAGS = flags.FLAGS


#make the log directory in which the log files for tensorboard are saved.
if not FLAGS.log_dir:
  log_dir = FLAGS.checkpoint_dir
else:
  log_dir = FLAGS.log_dir
if not os.path.exists(log_dir):
  os.makedirs(log_dir)

root_outfile_path = "%s/epochs_%s_to_%s.root" % (log_dir, FLAGS.EpochStart, FLAGS.Nepochs)


def main(_):
  pp.pprint("Arguments:")
  pp.pprint(flags.FLAGS.__flags)

  train = not FLAGS.finalSave
  BatchSize = 1 if (FLAGS.finalSave) else FLAGS.batch_size
  print("Batch size is", BatchSize)

  pp.pprint("[%s] Setup the graphs" % time.ctime())
  with tf.device("/gpu:%s" % FLAGS.GPUIndex if not FLAGS.finalSave else "/cpu"): 
    #define the placeholders
    x_images = tf.placeholder(tf.float32, [BatchSize] + [FLAGS.height, FLAGS.width, FLAGS.color_dim], name='real_images') #the real images
    z = tf.placeholder(tf.float32, [BatchSize, FLAGS.z_dim], name='z') #the input noise vector
    epsilon = tf.placeholder(tf.float32, [BatchSize, 1, 1, 1], name="epsilon")
    energy = tf.placeholder(tf.float32, [BatchSize, 1], name='energy_real')  #the input energy of the 
    impactPoint = tf.placeholder(tf.float32, [BatchSize, 2], name='impactPoint')  #the input point onto the first layer as measured by the dwcs 
    lambda_c = tf.placeholder(tf.float32, name='lambda_c')
    kappa_e_ph = tf.placeholder(tf.float32, name='kappa_e_ph')
    kappa_p_ph = tf.placeholder(tf.float32, name='kappa_p_ph')
    
    #import the models
    #generator
    import architectures.generators.G5 as generator_module  
    generator = generator_module.generic_generator(BatchSize, FLAGS.color_dim, train, False, "generator")(z, impactPoint, energy)

    #critic
    import architectures.critics.C1 as critic_module
    critic = critic_module.generic_critic(BatchSize, FLAGS.color_dim, train, False, "critic")(x_images, impactPoint, energy)
    _critic = critic_module.generic_critic(BatchSize, FLAGS.color_dim, train, True, "critic")(generator, impactPoint, energy)
    x_hat = epsilon * x_images + (1.0 - epsilon) * generator
    __critic = critic_module.generic_critic(BatchSize, FLAGS.color_dim, train, True, "critic")(x_hat, impactPoint, energy)
    critic_g = critic_module.generic_critic(BatchSize, FLAGS.color_dim, False, True, "critic")(generator, impactPoint, energy)

    #energy regressor
    import architectures.energy_regressors.e3 as energy_regressor_module    
    energy_regressor = energy_regressor_module.generic_energy_regressor(BatchSize, FLAGS.color_dim, train, False, "energy_regressor")(x_images)
    energy_regressor_cost = energy_regressor_module.generic_energy_regressor(BatchSize, FLAGS.color_dim, False, True, "energy_regressor")(x_images)
    energy_regressor_g = energy_regressor_module.generic_energy_regressor(BatchSize, FLAGS.color_dim, False, True, "energy_regressor")(generator)

    #position regressor
    import architectures.position_regressors.p2 as position_regressor_module      
    position_regressor = position_regressor_module.generic_position_regressor(BatchSize, FLAGS.color_dim, train, False, "position_regressor")(x_images)
    position_regressor_cost = position_regressor_module.generic_position_regressor(BatchSize, FLAGS.color_dim, False, True, "position_regressor")(x_images)
    position_regressor_g = position_regressor_module.generic_position_regressor(BatchSize, FLAGS.color_dim, False, True, "position_regressor")(generator)

    #add some book-keeping for the energy regressor network outputs
    e_sum = histogram_summary("energy_regressor_real", energy_regressor)
    e__sum = histogram_summary("energy_regressor_fakes", energy_regressor_g)
      
    #add some book-keeping for the position regressor network outputs
    p_sum = histogram_summary("position_regressor_real", position_regressor)
    p__sum = histogram_summary("position_regressor_fakes", position_regressor_g)


    #define error functions based on the wasserstein distance:
    c_loss_real = em_loss(tf.ones(BatchSize), critic)
    c_loss_fake = em_loss(tf.ones(BatchSize), _critic)

    # gradient penalty to guarantee 1-Lipschitz functions
    gradients_c = tf.gradients(__critic, [x_hat])
    gradient_penalty = tf.square(tf.norm(gradients_c[0], ord=2) - 1.0)
    c_loss = - (c_loss_real-c_loss_fake) + lambda_c*gradient_penalty

    #add the energy regression cost
    e_loss_real_min = tf.reduce_mean((energy_regressor-energy)**2)
    e_loss_real = tf.reduce_mean((energy_regressor_cost-energy)**2)
    e_loss_fake = tf.reduce_mean((energy_regressor_g-energy)**2) 
    e_loss_fake_min = tf.abs(e_loss_real-e_loss_fake)
    #add the position regression cost
    p_loss_real_min = tf.reduce_mean((position_regressor-impactPoint)**2)
    p_loss_real = tf.reduce_mean((position_regressor_cost-impactPoint)**2)
    p_loss_fake = tf.reduce_mean((position_regressor_g-impactPoint)**2) 
    p_loss_fake_min = tf.abs(p_loss_real-p_loss_fake)

    #minimize the critic network's output, this is the central trick on how to achieve the energy/position dependence and at the same time to fool the discriminator!
    g_loss = -em_loss(tf.ones(BatchSize), critic_g)
    g_loss_total = g_loss+kappa_e_ph*e_loss_fake_min+kappa_p_ph*p_loss_fake_min


    ##### book-keeping with tensorboards: #####

    #add some book-keeping for the loss functions
    c_loss_real_sum = scalar_summary("c_loss_real", c_loss_real)
    c_loss_fake_sum = scalar_summary("c_loss_fake", c_loss_fake)
    gradient_penalty_sum = scalar_summary("gradient_penalty", gradient_penalty)
    c_loss_sum = scalar_summary("c_loss", c_loss)

    g_loss_sum = scalar_summary("g_loss", g_loss)
    g_loss_total_sum = scalar_summary("g_loss_total", g_loss_total)
    e_loss_real_sum = scalar_summary("e_loss_real", e_loss_real)
    e_loss_fake_sum = scalar_summary("e_loss_fake", e_loss_fake)
    e_loss_fake_min_sum = scalar_summary("e_loss_fake_min", e_loss_fake_min)
    p_loss_real_sum = scalar_summary("p_loss_real", p_loss_real)
    p_loss_fake_sum = scalar_summary("p_loss_fake", p_loss_fake)
    p_loss_fake_min_sum = scalar_summary("p_loss_fake_min", p_loss_fake_min)

    #merge summary objects:
    g_sum = merge_summary([g_loss_total_sum, g_loss_sum])
    c_sum = merge_summary([c_loss_real_sum, c_loss_fake_sum, gradient_penalty_sum, c_loss_sum])
    e_sum_real = merge_summary([e_sum, e_loss_real_sum])
    e_sum_fakes = merge_summary([e__sum, e_loss_fake_sum])
    p_sum_real = merge_summary([p_sum, p_loss_real_sum])
    p_sum_fakes = merge_summary([p__sum, p_loss_fake_sum])

    all_sum = merge_summary([c_sum, g_sum, e_sum_real, e_sum_fakes, p_sum_real, p_sum_fakes])


  pp.pprint("[%s] Initialize the tensorflow session" % time.ctime())
  #configure tensorflow:

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

  #initialise save and log objects
  pp.pprint("[%s] Initialize the save object and logs" % time.ctime())
  saver = tf.train.Saver(max_to_keep=1000)
  
  if not FLAGS.finalSave:
    #read the data
    import helpers.data as data
    images_train,energies_train,dwcsReference_train,images_test,energies_test,dwcsReference_test = data.getData(FLAGS.input_dir, FLAGS.color_dim, load_evaluation=False)

    Nimages = len(images_train)
    Nimages_test = len(images_test)

    pp.pprint("[%s] %s images selected for training..." % (time.ctime(), Nimages))
    pp.pprint("[%s] %s images selected for testing..." % (time.ctime(), Nimages_test))

    #shuffle test sample
    perm_test = np.arange(Nimages_test)    
    np.random.shuffle(perm_test)
    images_test = images_test[perm_test] #shuffle
    energies_test = energies_test[perm_test] #shuffle
    dwcsReference_test = dwcsReference_test[perm_test]

    tf.placeholder(tf.float32, name='kappa_e_ph')
    learning_rateC = tf.placeholder(tf.float32, name='learning_rateC')
    learning_rateG = tf.placeholder(tf.float32, name='learning_rateG')
    learning_rateE = tf.placeholder(tf.float32, name='learning_rateE')
    learning_rateP = tf.placeholder(tf.float32, name='learning_rateP')
    beta1 = trainingParameters["beta1"]
    beta2 = trainingParameters["beta2"]

    batch_z_eval = np.random.uniform(-1., 1., [BatchSize, FLAGS.z_dim]).astype(np.float32)   #constant noise vector for evaluation

    #initialise the summary writer
    writer = SummaryWriter(log_dir, sess.graph)
    
    #start the training:
    start_time = time.time()
    pp.pprint("[%s] Defining the minimisation during the training..." % time.ctime())
    #define the minimisation methods
    t_vars = tf.trainable_variables()
    c_vars = [var for var in t_vars if 'c_' in var.name]
    c_optim = tf.train.AdamOptimizer(learning_rateC, beta1=beta1, beta2=beta2).minimize(c_loss, var_list=c_vars)

    g_vars = [var for var in t_vars if 'g_' in var.name]
    g_optim = tf.train.AdamOptimizer(learning_rateG, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)
    g_tot_optim = tf.train.AdamOptimizer(learning_rateG, beta1=beta1, beta2=beta2).minimize(g_loss_total, var_list=g_vars)

    e_vars = [var for var in t_vars if 'e_' in var.name]
    e_optim = tf.train.AdamOptimizer(learning_rateE) \
              .minimize(e_loss_real_min, var_list=e_vars)

    p_vars = [var for var in t_vars if 'p_' in var.name]
    p_optim = tf.train.AdamOptimizer(learning_rateP) \
              .minimize(p_loss_real_min, var_list=p_vars)
    
    pp.pprint("[%s] Initialize the variables" % time.ctime())
    tf.global_variables_initializer().run(session=sess)
    
    #loading the trained variables
    load_success, min_counter = load(saver, sess, FLAGS.checkpoint_dir, loadCounter=FLAGS.loadCounter)
    
    pp.pprint("[%s] Created root output file: %s" % (time.ctime(), root_outfile_path))
    root_outfile = ROOT.TFile(root_outfile_path, "RECREATE")
    root_outfile.Close()

    pp.pprint("[%s] Preparing the root TTree file." % time.ctime())
    tree_vars = ["epoch/I", "step/I", "batchsize/I", "DoCTrain/I", "DoGTrain/I", "trainGtotEvery/I", "beta1_GD/F", "beta2_GD/F"]
    tree_vars += ["G_Model/I", "C_Model/I", "E_Model/I", "P_Model/I"]
    tree_vars += ["DoETrain/I", "DoPTrain/I", "learning_rateC/F", "learning_rateG/F", "lambda/F", "kappa_e/F", "kappa_p/F", "errC_tot/F", "errC_tot_test/F"]
    tree_vars += ["errPenalty/F", "errG/F", "errG_discr/F", "errG_tot/F", "errE_real/F", "errE_real_test/F", "errE_fake/F", "errE_fake_min/F", "errP_real/F", "errP_real_test/F", "errP_fake/F", "errP_fake_min/F"]
    from helpers.tree_writer import tree_writer 
    outtree = tree_writer("GANTraining", tree_vars)
    
    pp.pprint("[%s] Starting the actual training" % time.ctime())
    N_batches = int(Nimages/BatchSize)
    if FLAGS.EpochStart > -1 or not load_success:
        epoch_start = FLAGS.EpochStart
    else:
        epoch_start = 1 + int(min_counter/N_batches)

    perm = np.arange(Nimages)

    for epoch in range(epoch_start, FLAGS.Nepochs+1):
      batch_images_test = images_test[0:BatchSize] 
      batch_energies_test = energies_test[0:BatchSize]
      batch_dwcsReference_test = dwcsReference_test[0:BatchSize]     

      start_time_epoch = time.time()

      #actual training only for positive lambdas
      if epoch > 0:
        Lambda = trainingParameters["Lambda"][epoch-1]
        kappa_e = trainingParameters["kappa_e"][epoch-1]
        kappa_p = trainingParameters["kappa_p"][epoch-1]
        _learning_rateC = trainingParameters["learning_rateC"][epoch-1]
        _learning_rateG = trainingParameters["learning_rateG"][epoch-1]
        _learning_rateE = trainingParameters["learning_rateE"][epoch-1]
        _learning_rateP = trainingParameters["learning_rateP"][epoch-1]

        DoCTrain = trainingParameters["DoCTrain"][epoch-1]
        DoEnergyRecoTrain = trainingParameters["DoEnergyRecoTrain"][epoch-1]
        DoPositionRecoTrain = trainingParameters["DoPositionRecoTrain"][epoch-1]
        DoGTrain = trainingParameters["DoGTrain"][epoch-1]
        trainGTotEvery = trainingParameters["trainGTotEvery"][epoch-1]
        np.random.shuffle(perm)
        images_train = images_train[perm] #shuffle
        energies_train = energies_train[perm] #shuffle
        dwcsReference_train = dwcsReference_train[perm]
        pp.pprint("[%s] Starting epoch %s" % (time.ctime(), epoch))

        #test_batch_index = 0
        for batch_idx in range(N_batches):
          min_counter+=1
          batch_images = images_train[batch_idx*BatchSize:(batch_idx+1)*BatchSize]
          batch_energies_real = energies_train[batch_idx*BatchSize:(batch_idx+1)*BatchSize]
          batch_dwcsReference_real = dwcsReference_train[batch_idx*BatchSize:(batch_idx+1)*BatchSize]
          

          batch_idx_test = batch_idx % (math.floor(Nimages_test/BatchSize)-1)
          index_start = int(batch_idx_test*BatchSize)
          index_end = int((batch_idx_test+1)*BatchSize)
          batch_images_test = images_test[index_start:index_end] 
          batch_energies_test = energies_test[index_start:index_end]
          batch_dwcsReference_test = dwcsReference_test[index_start:index_end]   


          batch_z = np.random.uniform(-1., 1., [BatchSize, FLAGS.z_dim]).astype(np.float32)
          batch_epsilon = np.random.uniform(0., 1., [BatchSize, 1, 1, 1]).astype(np.float32)

          # 1. Update G network
          if DoGTrain and not min_counter % trainGTotEvery and trainGTotEvery>-1:
            _ = sess.run(g_tot_optim, feed_dict={ x_images: batch_images, z: batch_z, energy: batch_energies_real, impactPoint: batch_dwcsReference_real, kappa_e_ph: kappa_e, kappa_p_ph: kappa_p, learning_rateG: _learning_rateG})
          else:
            # or 2. Update C network 
            if DoCTrain:
              _ = sess.run(c_optim, feed_dict={ x_images: batch_images, z: batch_z, epsilon: batch_epsilon, energy: batch_energies_real, impactPoint: batch_dwcsReference_real, lambda_c: Lambda, learning_rateC: _learning_rateC})
            
            #or 3. energy regression
            if DoEnergyRecoTrain:
              _ = sess.run(e_optim, feed_dict={ x_images: batch_images, z: batch_z, energy: batch_energies_real, impactPoint: batch_dwcsReference_real, learning_rateE: _learning_rateE})
                      
            #or 4. position regression
            if DoPositionRecoTrain:
              _ = sess.run(p_optim, feed_dict={ x_images: batch_images, z: batch_z, energy: batch_energies_real, impactPoint: batch_dwcsReference_real, learning_rateP: _learning_rateP})
            

          ###  BOOK keeping from here on ###

          #some book-keeping           with tensorboard
          summary_str = sess.run(all_sum, feed_dict={x_images: batch_images, z: batch_z, epsilon: batch_epsilon, energy: batch_energies_real, impactPoint: batch_dwcsReference_real, lambda_c: Lambda, kappa_e_ph: kappa_e, kappa_p_ph: kappa_p})
          writer.add_summary(summary_str, min_counter)

          #some book-keeping           with tensorboard
          errC_tot, errPenalty, errG, errG_tot, errE_real, errE_fake, errE_fake_min, errP_real, errP_fake, errP_fake_min = sess.run([c_loss, gradient_penalty, g_loss, g_loss_total, e_loss_real, e_loss_fake, e_loss_fake_min, p_loss_real, p_loss_fake, p_loss_fake_min], feed_dict={x_images: batch_images, z: batch_z, epsilon: batch_epsilon, energy: batch_energies_real, impactPoint: batch_dwcsReference_real, lambda_c: Lambda, kappa_e_ph: kappa_e, kappa_p_ph: kappa_p})
          errC_tot_test, errE_real_test, errP_real_test = sess.run([c_loss, e_loss_real, p_loss_real], feed_dict={x_images: batch_images_test, z: batch_z, epsilon: batch_epsilon, energy: batch_energies_test, impactPoint: batch_dwcsReference_test, lambda_c: Lambda, kappa_e_ph: kappa_e, kappa_p_ph: kappa_p})      
          
          #fill the output tree
          outtree({
            "epoch": epoch,
            "step": min_counter,
            "batchsize": BatchSize,
            "DoCTrain": int(DoCTrain),
            "DoGTrain": int(DoGTrain),
            "trainGtotEvery": trainGTotEvery,
            "beta1_GD": beta1,
            "beta2_GD": beta2,
            "DoETrain": int(DoEnergyRecoTrain),
            "DoPTrain": int(DoPositionRecoTrain),
            "learning_rateC": _learning_rateC,
            "learning_rateG": _learning_rateG,
            "G_Model": 5,
            "C_Model": 1,
            "E_Model": 3,
            "P_Model": 2,
            "lambda": Lambda,
            "kappa_e": kappa_e,
            "kappa_p": kappa_p,
            "errC_tot": errC_tot,
            "errC_tot_test": errC_tot_test, 
            "errPenalty": errPenalty, 
            "errG": errG, 
            "errG_tot": errG_tot, 
            "errE_real": errE_real, 
            "errE_real_test": errE_real_test, 
            "errE_fake": errE_fake, 
            "errE_fake_min": errE_fake_min,
            "errP_real": errP_real,
            "errP_real_test": errP_real_test,
            "errP_fake": errP_fake,
            "errP_fake_min": errP_fake_min
          })

          pp.pprint("[%s] Epoch: [%2d/%2d] [%4d/%4d] c_loss_tot: %.3f (%.3f) g_loss: %.3f, gradient_penalty; %.3f" \
            % (time.ctime(), epoch, FLAGS.Nepochs, batch_idx+1, N_batches, errC_tot, errC_tot_test, errG, errPenalty))
        
          pp.pprint("e_loss_real: %.3f (%.3f), e_loss_fake_min: %.3f, p_loss_real: %.3f (%.3f), p_loss_fake_min: %.3f" \
            % (errE_real, errE_real_test, errE_fake_min,errP_real, errP_real_test, errP_fake_min))
          #end of loop over the batches

        pp.pprint("Training across this epoch took %s minutes." % round((time.time() - start_time_epoch)/60., 2))
        save(saver, sess, FLAGS.checkpoint_dir, min_counter)
        #end of one epoch > 0


      ### Basic evaluation (testing) after each epoch ###

      #write performance indicators after each epoch
      root_outfile = ROOT.TFile(root_outfile_path, "UPDATE")
      outtree.write()
      epoch_dir = root_outfile.mkdir("gen_image_epoch%s"%epoch);
      epoch_dir.cd()

      pp.pprint("[%s] Evaluating a batch of images for visualisation" % time.ctime())
      #evaluate a full batch because batch norm is used
      batch_energy_eval = batch_energies_test
      batch_dwc_eval = batch_dwcsReference_test
      
      #indicate which energies are to be evaluated
      energies_for_eval = [20, 32, 50, 80, 90]
      for e in range(len(energies_for_eval)):
        _energy = energies_for_eval[e]*1.
        batch_energy_eval[e] = _energy
        batch_dwc_eval[e] = FLAGS.PositionForEval

      start_time_samplegen = time.time()
      gen_set = sess.run(generator, feed_dict={z: batch_z_eval, energy: batch_energy_eval, impactPoint: batch_dwc_eval})
      pp.pprint("Generation of %s samples took %s seconds." % (len(batch_energy_eval), round((time.time() - start_time_samplegen), 2)))
      
      for l in range(FLAGS.color_dim):
        layer_dir = epoch_dir.mkdir("layer_%s"%(l+1));
        layer_dir.cd()
        gen_pictures = []
        gen_pictures_average = []
        for e in range(len(energies_for_eval)):
          _energy = energies_for_eval[e]
          gen_picture_layer = ROOT.TH2F("energy_layer%s_%sGeV"%((l+1), _energy), "energy_layer%s_%sGeV"%((l+1), _energy), FLAGS.width, -0.5, FLAGS.width-0.5, FLAGS.height, -0.5, FLAGS.height-0.5)
          gen_picture_layer.GetXaxis().SetTitle("x")
          gen_picture_layer.GetYaxis().SetTitle("y")
          gen_picture_layer.GetZaxis().SetTitle("energy [MIP]")
          if e == len(energies_for_eval)-1:
            gen_picture_layer_average = ROOT.TH2F("average_energy_layer%s"%(l+1), "average_energy_layer%s"%(l+1), FLAGS.width, -0.5, FLAGS.width-0.5, FLAGS.height, -0.5, FLAGS.height-0.5)
            gen_picture_layer_average.GetXaxis().SetTitle("x")
            gen_picture_layer_average.GetYaxis().SetTitle("y")
            gen_picture_layer_average.GetZaxis().SetTitle("average energy [MIP]")
          for i in range(FLAGS.width):
            for j in range(FLAGS.height): 
              #transform intensities
              intensity = gen_set[e][j][i][l]
              if LogScale: 
                intensity = math.exp(intensity) - 1.
              if intensity <= GlobalNoiseThreshold:
                intensity = 0

              gen_picture_layer.Fill(i, j, intensity)
              if e == len(energies_for_eval)-1:
                for ev in range(4, len(batch_z_eval)):
                  _intensity = gen_set[ev][j][i][l]
                  if LogScale: 
                    _intensity = math.exp(_intensity) - 1.
                  if _intensity <= GlobalNoiseThreshold:
                    _intensity = 0
                  gen_picture_layer_average.Fill(i, j, gen_set[ev][j][i][l])
            
          gen_picture_layer.SetTitle("layer %s, impact [mm]: (%s,%s), energy: %s GeV" % ((l+1), FLAGS.PositionForEval, FLAGS.PositionForEval, _energy))
          gen_pictures.append(deepcopy(gen_picture_layer))
          gen_pictures[e].Write()
          
          if e == len(energies_for_eval)-1:
            gen_picture_layer_average.Scale(1./len(batch_z_eval))
            gen_picture_layer_average.SetTitle("layer %s, average" % ((l+1)))
            gen_pictures_average.append(deepcopy(gen_picture_layer_average))
            gen_pictures_average[-1].Write()
            
      epoch_dir.cd()
      rec_energies = sess.run(energy_regressor_g, feed_dict={ z: batch_z_eval, energy: batch_energy_eval, impactPoint:batch_dwc_eval  })
      pp.pprint("[%s] Creating distributions for energy reconstruction using fakes" % time.ctime())
      reco_energy_correlation = ROOT.TH2F("recoEnergy_epoch%s"%epoch, "recoEnergy_epoch%s"%epoch, int(np.max(batch_energies_test)-np.min(batch_energies_test)+1), np.min(batch_energies_test), np.max(batch_energies_test)+1, 100, np.min(rec_energies), np.max(rec_energies))
      for i in range(len(rec_energies)):
        reco_energy_correlation.Fill(batch_energies_test[i], rec_energies[i])
      reco_energy_correlation.GetXaxis().SetTitle("label")
      reco_energy_correlation.GetYaxis().SetTitle("reco")
      reco_energy_correlation.Write()
        
      rec_positions = sess.run(position_regressor_g, feed_dict={ z: batch_z_eval, energy: batch_energy_eval, impactPoint:batch_dwc_eval })
      pp.pprint("[%s] Creating test distributions for position reconstruction" % time.ctime())
      reco_positionX_correlation = ROOT.TH2F("recoPosX_epoch%s"%epoch, "recoPosX_epoch%s"%epoch, 50, np.min(batch_dwcsReference_test), np.max(batch_dwcsReference_test), 50, np.min(rec_positions), np.max(rec_positions))
      reco_positionY_correlation = ROOT.TH2F("recoPosY_epoch%s"%epoch, "recoPosY_epoch%s"%epoch, 50, np.min(batch_dwcsReference_test), np.max(batch_dwcsReference_test), 50, np.min(rec_positions), np.max(rec_positions))
      for i in range(len(rec_positions)):
        reco_positionX_correlation.Fill(batch_dwcsReference_test[i][0], rec_positions[i][0])
        reco_positionY_correlation.Fill(batch_dwcsReference_test[i][1], rec_positions[i][1])
      reco_positionX_correlation.GetXaxis().SetTitle("label")
      reco_positionX_correlation.GetYaxis().SetTitle("reco")
      reco_positionY_correlation.GetXaxis().SetTitle("label")
      reco_positionY_correlation.GetYaxis().SetTitle("reco")
      reco_positionX_correlation.Write()
      reco_positionY_correlation.Write()

      root_outfile.Close()
      pp.pprint("[%s] Updated root output file: %s" % (time.ctime(), root_outfile_path))
    
  
  elif FLAGS.finalSave:
    load_success, min_counter = load(saver, sess, FLAGS.checkpoint_dir, FLAGS.loadCounter)    
    #final saving
    finalDir = "%s/final" % FLAGS.checkpoint_dir
    builder = tf.saved_model.builder.SavedModelBuilder(finalDir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()
    pp.pprint("[%s] Final model saved in %s" % (time.ctime(), "%s" % finalDir))
  

if __name__ == '__main__':
  tf.app.run()