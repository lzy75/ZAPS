import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from utils import diag_ffhq_action_geometry_json_fix as fix
from utils import ffhq_action_geometry as rules


class SerializationTests(unittest.TestCase):
    def test_real_probe_record_strict_json_roundtrip(self):
        clean={g:np.full((1,3,2,2),g,dtype=np.float32) for g in rules.GAMMAS}
        response=rules.features(clean,clean,np.ones((1,3,2,2),np.float32),.05,{},repeat=clean[1.])
        record=dict(points=[dict(response=response)],extra=[np.bool_(True),np.int64(36),np.float32(.5)])
        plain=fix.native_scalars(record)
        decoded=json.loads(json.dumps(plain,allow_nan=False))
        self.assertIs(type(plain['points'][0]['response']['resolved']),bool)
        self.assertEqual(decoded['extra'],[True,36,.5])
        self.assertEqual(decoded['points'][0]['response']['active_measurements'],12)
        self.assertEqual(decoded['points'][0]['response']['resolved'],bool(response['resolved']))
        # Casting is restricted to JSON representation and does not mutate live measurements.
        self.assertIsInstance(record['extra'][0],np.bool_)

    def test_both_actual_write_aliases_signatures_and_frozen_bytes(self):
        source,scan=fix.original.source,fix.original.scan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source.signed(root/'plan.json',dict(worker_identity_files={}))
            frozen=(root/'plan.json').read_bytes();compat='a'*64
            with patch.object(source,'WRITE',source.WRITE),patch.object(scan,'WRITE',scan.WRITE):
                fix.install(root,compat)
                record=dict(plan_sha256=source.digest(root/'plan.json'),resolved=np.bool_(True),nfe=np.int64(36))
                folder=root/'case'/'RESPONSE'/'attempt_000';folder.mkdir(parents=True)
                scan.WRITE(folder/'run.json',record)
                observed=json.loads((folder/'run.json').read_text())
                self.assertIs(observed['resolved'],True)
                self.assertEqual(observed['serialization_compat_sha256'],compat)
                source.signed(folder/'run.json',record)
                self.assertEqual(source.load_signed(folder/'run.json'),observed)
                self.assertEqual(hashlib.sha256((folder/'run.json').read_bytes()).hexdigest(),
                                 (folder/'run.sha256').read_text().strip())
                with self.assertRaises(ValueError):scan.WRITE(root/'bad.json',dict(v=np.float64(np.nan)))
            self.assertEqual((root/'plan.json').read_bytes(),frozen)


if __name__=='__main__':unittest.main()
