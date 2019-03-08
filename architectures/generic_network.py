import tensorflow as tf
class generic_network_class(object):
  def __init__(self, batch_size, channels, trainable, reuse, scope_name="energy_regressor"):
    self.batch_size = batch_size
    self.channels = channels
    self.trainable = trainable
    self.reuse = reuse
    self.scope_name = scope_name
    self.initalized = False
  
  def __call__(self):
    raise NotImplementedError