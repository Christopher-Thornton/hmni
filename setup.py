from setuptools import setup, find_packages

setup_args = dict(
    name='hmni',
    version='0.1.1',
    description='Fuzzy Name Matching with Machine Learning',
    license='MIT',
    packages=find_packages(),
    author='Christopher Thornton',
    author_email='christopher_thornton@outlook.com',
    keywords=['Fuzzy', 'Name', 'Matching'],
    url='https://github.com/Christopher-Thornton/hmni',
    download_url='https://pypi.org/project/hmni/'
)

install_requires = [
    'tensorflow >= 1.11,< 2.0',
    'pandas >= 0.25',
    'numpy',
    'unidecode',
    'fuzzywuzzy',
    'abydos == 0.5.0',
    'sklearn',
    'nltk',
    'joblib',
    'six'
]


if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
