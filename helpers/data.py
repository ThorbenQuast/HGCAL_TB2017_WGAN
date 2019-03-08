import time
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
import os
import pdb

from helpers.config import GlobalNoiseThreshold, SignalSmearing, LogScale

#20GeV electrons
electron_runs = [1647, 1648]
#32GeV electrons
electron_runs += [1641, 1642]
#50GeV electrons
electron_runs += [1634, 1639]
#80GeV electrons
electron_runs += [1632, 1637]
#90GeV electrons
electron_runs += [1651, 1652]


electrons_sim = ["reco_config2_pdgID11_beamMomentum20_listFTFP_BERT_EML.npz"]
electrons_sim += ["reco_config2_pdgID11_beamMomentum32_listFTFP_BERT_EML.npz"]
electrons_sim += ["reco_config2_pdgID11_beamMomentum50_listFTFP_BERT_EML.npz"]
electrons_sim += ["reco_config2_pdgID11_beamMomentum80_listFTFP_BERT_EML.npz"]
electrons_sim += ["reco_config2_pdgID11_beamMomentum90_listFTFP_BERT_EML.npz"]

electrons_sim_eval = ["reco_config2_pdgID11_beamMomentum70_listFTFP_BERT_EML.npz"]


def getData(input_dir, channels, load_evaluation=False):
  np.random.seed(seed=0)  #fixed seed for reproducability
  NTrainFraction = 0. if load_evaluation else 0.9 

  simFiles = electrons_sim
  if load_evaluation:
    simFiles += electrons_sim_eval 
  filepaths = ["%s/%s" % (input_dir, file) for file in simFiles]

  #load the real images
  for i in range(len(filepaths)):
    filepath = filepaths[i]
    dataFile = np.load(filepath)
    pp.pprint("[%s] Loading data file %s" % (time.ctime(), filepath))
    

    images = dataFile["rechits"][:, 0:channels,:, :, 0:1]
    Nimages = len(images)
    pp.pprint("[%s] %s images loaded..." % (time.ctime(), Nimages))
    energies = dataFile["event"][:, 3:4]
    dwcsTrackType = dataFile["dwcReference"][:, 0, 0]
    dwcsReference = dataFile["dwcReference"][:, 1, 0:2]
    dwcsChi2X = dataFile["dwcReference"][:, -1, 0]
    dwcsChi2Y = dataFile["dwcReference"][:, -1, 1]

    images = np.transpose(np.squeeze(images, axis=4), (0, 2, 3, 1))  #directly preprocess
    pp.pprint("[%s] %s images squeezed and transposed..." % (time.ctime(), Nimages))

    #selection into train and test samples
    selected_indexes = np.where((dwcsTrackType>=13) & (dwcsChi2X<5.) & (dwcsChi2Y<5.))[0]
    N_selected = len(selected_indexes)
    train_indexes = selected_indexes[0:int(NTrainFraction*N_selected)]
    test_indexes = selected_indexes[int(NTrainFraction*N_selected):-1]

    images_train = images[train_indexes]
    energies_train = energies[train_indexes]
    dwcsReference_train = dwcsReference[train_indexes]  

    images_test = images[test_indexes]
    energies_test = energies[test_indexes]
    dwcsReference_test = dwcsReference[test_indexes]
    pp.pprint("[%s] %s images selected: %s for training, %s for testing..." % (time.ctime(), N_selected, len(images_train), len(images_test)))

    if i==0:
      full_images_train = images_train
      full_energies_train = energies_train
      full_dwcsReference_train = dwcsReference_train
      full_images_test = images_test
      full_energies_test = energies_test
      full_dwcsReference_test = dwcsReference_test    
    else:
      full_images_train = np.concatenate((full_images_train, images_train), axis=0)
      full_energies_train = np.concatenate((full_energies_train, energies_train), axis=0)
      full_dwcsReference_train = np.concatenate((full_dwcsReference_train, dwcsReference_train), axis=0)
      full_images_test = np.concatenate((full_images_test, images_test), axis=0)
      full_energies_test = np.concatenate((full_energies_test, energies_test), axis=0)
      full_dwcsReference_test = np.concatenate((full_dwcsReference_test, dwcsReference_test), axis=0)  


  print("Shuffling training sample")
  perm = np.arange(len(full_images_train))
  np.random.shuffle(perm)
  full_images_train = full_images_train[perm]
  full_energies_train = full_energies_train[perm]
  full_dwcsReference_train = full_dwcsReference_train[perm]
  print("Shuffling test sample")
  perm = np.arange(len(full_images_test))
  np.random.shuffle(perm)
  full_images_test = full_images_test[perm]
  full_energies_test = full_energies_test[perm]
  full_dwcsReference_test = full_dwcsReference_test[perm]  


  if SignalSmearing > 0:
    print("Adding the smearing of %s MIPs to the training sample" % SignalSmearing)
    full_images_train = (full_images_train+np.random.normal(0., SignalSmearing, full_images_train.shape))
    print("Adding the smearing of %s MIPs to the testing sample" % SignalSmearing)
    full_images_test = (full_images_test+np.random.normal(0., SignalSmearing, full_images_test.shape))
  print("Perform clipping at noise cut of %s MIPs" % GlobalNoiseThreshold)
  full_images_train[full_images_train < GlobalNoiseThreshold] = 0.
  
  print("Perform clipping at noise cut of %s MIPs" % GlobalNoiseThreshold)
  

  full_images_test[full_images_test < GlobalNoiseThreshold] = 0.
  if LogScale:
    print("Transform train sample into logarithmic scale")
    full_images_train = np.log(1. + full_images_train)

    print("Transform testing sample into logarithmic scale")
    full_images_test = np.log(1. + full_images_test)  


  return full_images_train, full_energies_train, full_dwcsReference_train, full_images_test, full_energies_test, full_dwcsReference_test