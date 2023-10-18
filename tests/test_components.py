from itwinai.components import Executor


def test_slice():
    p = Executor(['step1', 'step2', 'step3'], pippo=2)
    assert len(p[:1]) == 1
    assert p[:1][0] == 'step1'
    assert len(p[1:]) == 2
    assert p[1:].constructor_args['pippo'] == 2
