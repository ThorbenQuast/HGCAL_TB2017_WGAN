import ROOT
from array import array

class tree_writer(object):
  def __init__(self, tree_name, branch_list):
    self.tree_name = tree_name
    self.tree = ROOT.TTree( self.tree_name , self.tree_name)
    self.buffer = {}
    for branch in branch_list:
      key = branch.split("/")[0]
      data_type = branch.split("/")[1].lower()
      self.buffer[key] = array( data_type, [ 0 ] )
      self.tree.Branch( key, self.buffer[key], branch )

  def __call__(self, values):
    for key in values:
      self.buffer[key][0] = values[key]

    self.tree.Fill()

  def write(self):
    self.tree.Write()