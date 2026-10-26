import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from package_predictions import sanitize,prediction_digest,IMAGE_KEYS,TOP_KEYS


class ReleaseSchemaTests(unittest.TestCase):
    def fixture(self):
        return {'schema_version':1,'selected_checkpoint':'last','checkpoint_sha256':'synthetic-hash',
          'FP32':True,'TTA':False,'confidence_floor':.001,'max_det':300,
          'purpose':'legacy metadata to remove','images':[{'id':'05-15_example','file_name':'05-15_example.png',
          'width':32,'height':24,'group_id':'05-15','gt_yolo':'legacy_label_reference.txt',
          'image_path':'legacy/input.png','auxiliary_beet_ground_truth':[{'class_id':0,'xyxy':[1,1,2,2]}],
          'predictions':[{'xyxy':[4.125,5.25,12.5,15.75],'class_id':1,'confidence':.0025}]}]}

    def test_whitelist_removes_gt_and_preserves_prediction_values(self):
        raw=self.fixture();before=copy.deepcopy(raw);clean=sanitize(raw)
        self.assertEqual(raw,before)
        self.assertEqual(set(clean),set(TOP_KEYS)|{'images'})
        self.assertEqual(set(clean['images'][0]),set(IMAGE_KEYS))
        self.assertEqual(clean['images'][0]['predictions'],raw['images'][0]['predictions'])
        self.assertEqual(prediction_digest(raw['images']),prediction_digest(clean['images']))

    def test_paths_and_nonprediction_payload_rejected(self):
        raw=self.fixture();raw['images'][0]['file_name']='nested/image.png'
        with self.assertRaises(AssertionError):sanitize(raw)
        raw=self.fixture();raw['images'][0]['predictions'][0]['ground_truth']=True
        with self.assertRaises(AssertionError):sanitize(raw)

    def test_export_floor_and_geometry_not_silently_changed(self):
        raw=self.fixture();raw['images'][0]['predictions'][0]['confidence']=.0001
        with self.assertRaises(AssertionError):sanitize(raw)
        raw=self.fixture();raw['images'][0]['predictions'][0]['xyxy'][2]=40
        with self.assertRaises(AssertionError):sanitize(raw)


if __name__=='__main__':unittest.main()
