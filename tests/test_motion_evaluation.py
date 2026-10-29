import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_g1_motion_acquisition import summarize


class MotionEvaluationTests(unittest.TestCase):
  def trace(self,velocity):
    n=len(velocity);t=np.arange(1,n+1)*.02
    return dict(time=t,velocity=velocity,position=np.cumsum(velocity,axis=0)*.02,
                rpy=np.zeros((n,3)),action=np.zeros((n,29)),contact_time=np.ones((n,2)),foot_velocity=np.zeros((n,2,2)))

  def test_sustained_forward(self):
    v=np.zeros((600,3));v[100:,0]=.2
    result=summarize(self.trace(v),(.2,0,0),False)
    self.assertTrue(result['acquired']);self.assertAlmostEqual(result['stable_actual_velocity'][0],.2)

  def test_transient_and_oscillation_are_not_acquisition(self):
    v=np.zeros((600,3));v[100:150,0]=.4
    self.assertFalse(summarize(self.trace(v),(.2,0,0),False)['acquired'])
    v[:,0]=.2*np.sin(np.arange(600)*.02*2*np.pi)
    self.assertFalse(summarize(self.trace(v),(.2,0,0),False)['acquired'])

  def test_wrong_direction_and_fall(self):
    v=np.zeros((600,3));v[100:,0]=-.2
    self.assertFalse(summarize(self.trace(v),(.2,0,0),False)['acquired'])
    v[:,0]=.2
    self.assertFalse(summarize(self.trace(v),(.2,0,0),True)['acquired'])
    result=summarize(self.trace(v[:40]),(.2,0,0),True)
    self.assertIsNone(result['stable_actual_velocity']);self.assertFalse(result['acquired'])


if __name__=='__main__':unittest.main()
