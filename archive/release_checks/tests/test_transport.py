"""Offline rendering and input guards; explicit geometric fixtures, never model runs."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from paper.transport import render_samples,validate_saved


class TransportTests(unittest.TestCase):
    def test_saved_source_checkpoint_and_scope_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.msgpack';path.write_bytes(b'saved state')
            record={'record_type':'paper_experiment','status':'complete','selection':{'manifold':'S2','method':'ours'},
                'checkpoint':{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':11},'source':{'sha256':'source'},'paper_source':{}}
            runtime=SimpleNamespace(source_manifest=lambda _:record['source'],__file__=__file__)
            validate_saved(record,path,Path(directory),runtime)
            record['selection']['manifold']='S10'
            with self.assertRaises(ValueError):validate_saved(record,path,Path(directory),runtime)
            record['selection']['manifold']='S2';path.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_saved(record,path,Path(directory),runtime)

    def test_offline_native_coordinate_render_and_metadata_for_sphere_and_torus(self):
        import numpy as np
        for manifold in ('S2','T2'):
            angles=np.linspace(0,2*np.pi,24,endpoint=False)
            points=np.stack([np.cos(angles),np.sin(angles),np.zeros_like(angles)],axis=1)
            if manifold=='T2':points=np.stack([np.cos(angles),np.sin(angles),np.cos(angles*.5),np.sin(angles*.5)],axis=1)
            arrays={key:points for key in ('source_samples','target_samples','transported_samples')}
            with tempfile.TemporaryDirectory() as directory:
                metadata={'manifold':manifold,'label':'TEST FIXTURE: identity coordinates, no trained model','smoke':True}
                result=render_samples(arrays,metadata,directory)
                self.assertEqual(result['arrays']['source_samples']['shape'],list(points.shape))
                self.assertEqual(result['input_metadata'],metadata)
                for name in ('transport.png','transport.pdf','transport.json'):self.assertGreater((Path(directory)/name).stat().st_size,100)

    def test_nonfinite_or_unpaired_samples_are_rejected(self):
        import numpy as np
        points=np.array([[1.,0.,0.],[0.,1.,0.]])
        arrays={key:points.copy() for key in ('source_samples','target_samples','transported_samples')}
        with tempfile.TemporaryDirectory() as directory:
            arrays['transported_samples'][0,0]=np.nan
            with self.assertRaises(ValueError):render_samples(arrays,{'manifold':'S2'},directory)
            arrays['transported_samples']=points[:1]
            with self.assertRaises(ValueError):render_samples(arrays,{'manifold':'S2'},directory)


if __name__=='__main__':unittest.main()
