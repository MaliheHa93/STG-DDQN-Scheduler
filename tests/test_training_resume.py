from pathlib import Path
import tempfile
import unittest
import numpy as np
from stg_ddqn.config import load_config
from stg_ddqn.environment import STGEnvironment
from stg_ddqn.neural import TORCH_AVAILABLE

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE,'PyTorch is optional for core simulator tests')
class TrainingResumeTests(unittest.TestCase):
    def test_mid_episode_restore_preserves_training_and_scientific_outcomes(self):
        import torch
        from stg_ddqn.neural import STGDDQNPolicy
        torch.set_num_threads(1)
        cfg=load_config(ROOT/'configs/smoke.yaml')
        cfg['simulation'].update(warmup_s=0,measurement_s=18)
        cfg['traffic']['arrival_rate_req_s']=2
        cfg['learning'].update(replay_warmup=4,batch_size=4,device='cpu')
        environment=STGEnvironment(cfg,11)
        policy=STGDDQNPolicy(cfg,seed=41)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt'
            torch.save(dict(environment=environment,policy=policy,rng=torch.get_rng_state()),path)
            reference,records=environment.run(policy,training=True)
            expected_weights={k:v.clone() for k,v in policy.online.state_dict().items()}
            initial=torch.load(path,weights_only=False)
            torch.set_rng_state(initial['rng'])
            class Interrupt(Exception): pass
            def capture(env,next_frame,total):
                if next_frame==7:
                    torch.save(dict(environment=env,policy=env.policy,rng=torch.get_rng_state()),path)
                    raise Interrupt
            with self.assertRaises(Interrupt):
                initial['environment'].run(initial['policy'],training=True,epoch_callback=capture)
            restored=torch.load(path,weights_only=False)
            torch.set_rng_state(restored['rng'])
            actual,actual_records=restored['environment'].run(restored['policy'],training=True,start_frame=7)
            self.assertEqual(records,actual_records)
            for key in reference:
                if 'runtime' in key: continue
                if isinstance(reference[key],float) and np.isnan(reference[key]):
                    self.assertTrue(np.isnan(actual[key]))
                else:
                    self.assertEqual(reference[key],actual[key],key)
            for key,value in restored['policy'].online.state_dict().items():
                torch.testing.assert_close(value,expected_weights[key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
