import hmni
import pandas as pd
from pandas._testing import assert_frame_equal

matcher = hmni.Matcher(model='latin')


# similarity
def test_similarity():
    assert matcher.similarity('Alan', 'Al') == 0.6838303319889133
    assert matcher.similarity('Alan', 'Al', prob=False) == 1
    assert matcher.similarity('Alan', 'Aiden', prob=False) == 0


# fuzzymerge
def test_fuzzymerge():
    df1 = pd.DataFrame({'name': ['Al', 'Mark', 'James', 'Harold']})
    df2 = pd.DataFrame({'name': ['Mark', 'Alan', 'James', 'Harold']})
    assert_frame_equal(matcher.fuzzymerge(df1, df2, how='left', on='name', limit=2), pd.DataFrame(
        {'name_x': ['Al', 'Mark', 'James', 'Harold'], 'name_y': ['Alan', 'Mark', 'James', 'Harold']}))


# dedupe
def test_dedupe():
    names = ['Alan', 'Al', 'Al', 'James']
    assert matcher.dedupe(names, keep='longest') == ['Alan', 'James']
    assert matcher.dedupe(names, keep='frequent') == ['Al', 'James']
    assert matcher.dedupe(names, keep='longest', replace=True) == ['Alan', 'Alan', 'Alan', 'James']


if __name__ == "__main__":
    test_similarity()
    test_fuzzymerge()
    test_dedupe()
    print("Everything passed")
