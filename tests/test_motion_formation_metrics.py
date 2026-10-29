import unittest
import numpy as np
from src.tasks.velocity.mdp.motion_formation_metrics import motion_formation_diagnostics


class FormationMetricTests(unittest.TestCase):
  def test_path_is_not_net_motion_and_action_variance_is_temporal(self):
    t=np.arange(1,601)*.02
    p=np.zeros((600,3));p[:,0]=np.sin(2*np.pi*t)
    a=np.ones((600,29))*2
    trace=dict(time=t,position=p,rpy=np.zeros((600,3)),action=a,
      contact_time=np.ones((600,2)),foot_velocity=np.zeros((600,2,2)),
      foot_position=np.zeros((600,2,3)))
    result=motion_formation_diagnostics(trace,True)['stable']
    self.assertGreater(result['root_horizontal_path'],10.)
    self.assertLess(abs(result['root_displacement_xyz'][0]),.2)
    self.assertEqual(result['action_rms'],2.)
    self.assertEqual(result['action_temporal_variance'],0.)
    self.assertEqual(result['double_support_fraction'],1.)
    self.assertEqual(result['foot_horizontal_displacement'],[[0.,0.],[0.,0.]])
    self.assertIsNone(motion_formation_diagnostics(trace,False)['stable'])


if __name__=='__main__':unittest.main()
