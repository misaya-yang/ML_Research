import unittest
from localization import operation_point, traced_worker
class Test(unittest.TestCase):
    def test_operations(self):
        s = 'def f(x):\n    """doc"""\n    # comment\n    return x+1 # trailing\n'
        self.assertFalse(operation_point(s,2,5))
        self.assertFalse(operation_point(s,3,5))
        self.assertTrue(operation_point(s,4,11))
        self.assertFalse(operation_point(s,4,19))
        self.assertFalse(operation_point('def f(:',1,0))
    def test_worker(self):
        from pathlib import Path
        import ast
        f=Path('results/local_runs/native-firstpass-1qr7b1va/score_cpython.py')
        source=ast.parse(f.read_text())
        w=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(getattr(x,'id','')=='WORKER' for x in n.targets))
        compile(traced_worker(w),'<worker>','exec')
if __name__=='__main__':unittest.main()
