"""CPU-only checks for joint PA tuning policy and cache-only resolution."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1] / "aiter/ops/flydsl"
spec = importlib.util.spec_from_file_location("pa_config_under_test", ROOT / "pa_decode_config.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def tuning_function(name, **namespace):
    source = ROOT / "pa_decode_tuning.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.unparse(fn), str(source), "exec"), namespace)
    return namespace[name]


class TestPAConfig(unittest.TestCase):
    def geometry(self, batch=12, **changes):
        values = dict(arch="gfx942", num_seqs=batch, num_kv_heads=1,
            query_group_size=16, head_dim=192, value_dim=128, block_size=64,
            query_length=8, query_dtype="torch.bfloat16", kv_dtype="torch.bfloat16",
            per_token_kv=False, trans_v=True, sink_dtype=None, sliding_window=0,
            max_partitions=304, num_cu=304, requested_query_splits=None)
        values.update(changes)
        return values

    def test_selected_serving_defaults(self):
        for batch, expected in [(12,(24,1)),(24,(64,2)),(40,(64,2)),(96,(3,1)),
                                (152,(2,1)),(208,(8,2)),(240,(4,2)),(256,(4,2))]:
            with self.subTest(batch=batch):
                result = policy.calibrated_pa_config(**self.geometry(batch))
                self.assertEqual((result['partition_count'],result['query_splits']),expected)
                self.assertEqual(result['workgroup_budget'],batch*expected[0])

    def test_untuned_geometry_and_explicit_cap(self):
        for changes in [dict(arch="gfx950"),dict(num_cu=256),
                        dict(query_length=4),dict(sliding_window=128),dict(sink_dtype="torch.float32"),
                        dict(kv_dtype="torch.float8_e4m3fnuz"),dict(num_kv_heads=2),
                        dict(trans_v=False)]:
            with self.subTest(changes=changes):
                self.assertIsNone(policy.calibrated_pa_config(**self.geometry(**changes)))
        for changes, parts in [(dict(num_seqs=13),16),(dict(max_partitions=16),16),
                                (dict(max_partitions=8),8)]:
            values=self.geometry(**changes)
            result=policy.calibrated_pa_config(**values)
            self.assertEqual(result,dict(partition_count=parts,query_splits=None,
                                        workgroup_budget=values['num_seqs']*parts))

    def test_explicit_split(self):
        result = policy.calibrated_pa_config(**self.geometry(requested_query_splits=2))
        self.assertEqual(result['query_splits'],2)

    def test_old_budget_cache_compatibility(self):
        self.assertEqual(policy.validate_pa_config({'workgroup_budget':608},
            max_partitions=304,query_length=8),(608,304,None))

    def test_invalid_cache_values(self):
        bad = [{}, {'workgroup_budget':True}, {'workgroup_budget':0},
            {'workgroup_budget':608,'partition_count':305},
            {'workgroup_budget':608,'partition_count':False},
            {'workgroup_budget':608,'query_splits':3},
            {'workgroup_budget':608,'query_splits':True},
            {'workgroup_budget':608,'unknown':1}]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                policy.validate_pa_config(values,max_partitions=304,query_length=8)
        with self.assertRaises(ValueError):
            policy.validate_pa_config({'workgroup_budget':288,'query_splits':2},
                max_partitions=304,query_length=8,requested_query_splits=1)

    def test_joint_candidates_are_not_deduplicated_by_capacity(self):
        prepare_calls=[]
        resources=SimpleNamespace(args=(object(),object(),SimpleNamespace(numel=lambda:12),None,None,8),
            options={'query_splits':None},max_partitions=128,prepare=lambda *a:prepare_calls.append(a))
        fn=tuning_function('_pa_decode_configs',Config=Config,_cache_geometry=lambda key:(1,64))
        results=[x.kwargs for x in fn(resources=resources,num_cu=304)]
        self.assertIn(dict(workgroup_budget=288,partition_count=24,query_splits=1),results)
        self.assertIn(dict(workgroup_budget=288,partition_count=24,query_splits=2),results)
        self.assertTrue(prepare_calls)
        resources.options['query_splits']=1
        self.assertTrue(all(x.kwargs['query_splits']==1 for x in fn(resources=resources,num_cu=304)))

    def test_cache_only_does_not_search(self):
        fn=tuning_function('_resolve_pa_decode_config')
        cached=Config(workgroup_budget=288,partition_count=24,query_splits=1)
        calls=[]
        tuner=SimpleNamespace(_make_key=lambda *a:'key',cache={'key':cached},
            _artifact_ref=lambda *a,**kw:None,_load_artifact=lambda *a:None,
            default=lambda **kw:calls.append('default') or cached,
            resolve_config=lambda **kw:(_ for _ in ()).throw(AssertionError('Search called')))
        self.assertIs(fn(tuner,cache_only=True),cached)
        self.assertEqual(calls,[])
        tuner.cache.clear()
        self.assertIs(fn(tuner,cache_only=True),cached)
        self.assertEqual(calls,['default'])


if __name__ == '__main__':
    unittest.main()
