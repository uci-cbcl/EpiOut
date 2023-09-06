from setuptools import setup, find_packages


with open('README.md') as readme_file:
    readme = readme_file.read()


requirements = [
    'setuptools',
    'tqdm',
    'pyyaml',
    'click',
    'numpy<=1.23',
    'pooch',
    'anndata',
    'pandas',
    'pyarrow',
    'pyranges',
    'scipy',
    'matplotlib',
    'seaborn',
    'pysam',
    'joblib',
    'scikit-learn',
    'pyBigWig',
    'statsmodels',
    'tensorflow',
    'tensorflow-probability'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest']

setup(
    name='epiout',
    version='0.0.1',

    author="M. Hasan Çelik",
    author_email='muhammedhasancelik@gmail.com',
    url='https://github.com/muhammedhasan/epiout',

    keywords=['genomics', 'ATAC-seq'],
    description="EpiOut: outlier detection for DNA accesibility data.",

    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    license="MIT license",
    long_description=readme + '\n',
    long_description_content_type='text/markdown',

    install_requires=requirements,
    setup_requires=setup_requirements,

    entry_points='''
        [console_scripts]
        epicount=epiout.main:cli_epicount
        epiout=epiout.main:cli_epiout
        epiannot=epiout.main:cli_epiannot
        epiannot_create=epiout.main:cli_epiannot_create
        epiannot_list=epiout.main:cli_epiannot_list
    ''',
    packages=find_packages(include=['epiout*']),
    include_package_data=True,

    test_suite='tests',
    tests_require=test_requirements,
)
