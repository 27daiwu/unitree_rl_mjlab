import unittest
import importlib.util
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_g1_flat_control import support_switches

class SupportTests(unittest.TestCase):
  def test_rocking_and_contact_chatter_do_not_count(self):
    t=np.arange(250)*.02+7.02
    self.assertFalse(support_switches(t,np.ones((250,2),dtype=bool),True)['alternating_support_acquired'])
    c=np.array([[True,False],[False,True]]*125)
    self.assertFalse(support_switches(t,c,True)['alternating_support_acquired'])
  def test_sustained_alternating_support_and_incomplete(self):
    c=np.repeat([[True,False],[True,True],[False,True],[True,True]]*3,10,axis=0)
    t=np.arange(len(c))*.02+7.02
    self.assertTrue(support_switches(t,c,True)['alternating_support_acquired'])
    self.assertFalse(support_switches(t,c,False)['alternating_support_acquired'])
