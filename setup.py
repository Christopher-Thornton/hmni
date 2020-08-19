from setuptools import setup

with open('README.md', encoding='utf8') as f:
    readme = f.read()

setup_args = dict(
    name='hmni',
    version='0.1.5',
    description='Fuzzy Name Matching with Machine Learning',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=['hmni', 'hmni.models'],
    include_package_data=True,
    author='Christopher Thornton',
    author_email='christopher_thornton@outlook.com',
    keywords=['fuzzy-matching', 'natural-language-processing', 'nlp', 'machine-learning', 'data-science', 'python', 'artificial-intelligence', 'ai'],
    url='https://github.com/Christopher-Thornton/hmni',
    download_url='https://github.com/Christopher-Thornton/hmni/archive/v0.1.5.zip',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
  ],
)

install_requires = [
    'tensorflow >= 1.11,< 2.0',
    'pandas >= 0.25',
    'numpy',
    'unidecode',
    'fuzzywuzzy',
    'abydos == 0.5.0',
    'sklearn',
    'joblib',
    'six'
]


if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
