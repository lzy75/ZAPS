import unittest
from utils import response_reliability as rules
from utils.diag_cached_reliability_audit import analyse


def record():
    values={g:dict(loss=10.+(g-6.)**2,risks={}) for g in (0.,.5,1.,1.5,2.,3.,4.,5.)}
    values[3.]['loss']=35.-.9*16.;values[4.]['loss']=35.-.7*21.;values[5.]['loss']=20.
    choice=rules.choose(values,5.,0.)
    return dict(case=dict(task='gaussian_deblur',key='synthetic',image_id='synthetic'),
        query_readouts={str(g):v for g,v in values.items()},decision=choice,
        gamma=dict(LEGACY=5.,**choice['gamma']),geometry={},
        evaluation={'LOCAL':{'metrics':dict(psnr=20.,ssim=.7,lpips=.2,loss=.1)},
            'LEGACY':{'metrics':dict(psnr=21.,ssim=.8,lpips=.1,loss=.2)}})


class AuditTests(unittest.TestCase):
    def test_detects_label_free_replay_without_equating_loss_with_quality(self):
        r=record();out=analyse([r],{'gaussian_deblur':0.})
        self.assertEqual(out['groups'][0]['trust_changed_from_local'],1)
        self.assertEqual(out['groups'][0]['mean_LOCAL_minus_LEGACY']['psnr'],-1.)
        self.assertEqual(out['groups'][0]['local_J_improved_final_PSNR_worse'],0)

    def test_action_and_decision_tampering_are_rejected(self):
        r=record();r['decision']['gamma']['TRUST']=99.
        with self.assertRaisesRegex(ValueError,'replay differs'):analyse([r],{'gaussian_deblur':0.})
        r=record();r['gamma']['TRUST']=99.
        with self.assertRaisesRegex(ValueError,'Signed action'):analyse([r],{'gaussian_deblur':0.})


if __name__=='__main__':unittest.main()
