import hmni

test_cases = {
    'exact_fullname': ('Alan Turing', 'Alan Turing'),
    'exact_fname': ('Alan', 'Alan'),

    'diff_fname': ('Alan Turing', 'Aiden Turing'),
    'diff_lname': ('Alan Turing', 'Alan Turbo'),
    'diff_all': ('Alan Turing', 'Joe Smith'),

    'filter_fname': ('Alan Turing', 'Bob Turing'),

    'alt_fname': ('Alan Turing', 'Al Turing'),
    'alt_lname': ('Alan Turing', 'Al Tering'),

    'initial_fname': ('Alan Turing', 'A Turing'),
    'initial_lname': ('Alan Turing', 'Alan T'),
    'initial_both': ('Alan Turing', 'A T'),

    'diff_initial_fname': ('A Turing', 'J Turing'),
    'diff_initial_lname': ('Alan T', 'Alan J'),
    'diff_initial_fname_pair': ('A T', 'J T'),
    'diff_initial_lname_pair': ('A T', 'A J'),
    'diff_initial_both_pair': ('A T', 'J J'),

    'missing_fname': ('Alan Turing', 'Turing'),
    'missing_lname': ('Alan Turing', 'Alan'),
}

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=True,
                       allow_initials=True, allow_missing_components=True)
print('\n0. similarity scores')
for k, v in test_cases.items():
    print(k, matcher.similarity(*v))

print([round(matcher.similarity(*v), 4) for v in test_cases.values()], '\n')

matcher = hmni.Matcher(model='latin', prefilter=True, allow_alt_surname=True,
                       allow_initials=True, allow_missing_components=True)
print('1. allow all')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0, 0.6838, 0.6838, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5]

matcher = hmni.Matcher(model='latin', prefilter=True, allow_alt_surname=True,
                       allow_initials=True, allow_missing_components=False)
print('2. no missing components')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0, 0.6838, 0.6838, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=True, allow_alt_surname=True,
                       allow_initials=False, allow_missing_components=False)
print('3. no missing components & no initials')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0, 0.6838, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=True, allow_alt_surname=False,
                       allow_initials=False, allow_missing_components=False)
print('4. no missing components & no initials & no alt surname')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=False,
                       allow_initials=False, allow_missing_components=False)
print('5. disallow all')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=True,
                       allow_initials=True, allow_missing_components=True)
print('6. no prefilter')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0.6838, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=False,
                       allow_initials=True, allow_missing_components=True)
print('7. no prefilter & no alt surname')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=False,
                       allow_initials=False, allow_missing_components=True)
print('8. no prefilter & no alt surname & no initials')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=True,
                       allow_initials=True, allow_missing_components=False)
print('9. no prefilter & no missing components')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0.6838, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=True,
                       allow_initials=False, allow_missing_components=False)
print('10. no prefilter & no missing components & no initials')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=False,
                       allow_initials=True, allow_missing_components=False)
print('11. no prefilter & no missing components & no alt surname')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=True, allow_alt_surname=False,
                       allow_initials=True, allow_missing_components=False)
print('12. no alt surname & no missing components')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0, 0.6838, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]

matcher = hmni.Matcher(model='latin', prefilter=False, allow_alt_surname=True,
                       allow_initials=False, allow_missing_components=True)
print('13. no prefilter & no initials')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0.0113, 0.6838, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]

matcher = hmni.Matcher(model='latin', prefilter=True, allow_alt_surname=False,
                       allow_initials=False, allow_missing_components=True)
print('14. no alt surname & no initials')
assert [round(matcher.similarity(*v), 4) for v in test_cases.values()] == \
       [1, 1, 0.2667, 0, 0, 0, 0.6838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]
