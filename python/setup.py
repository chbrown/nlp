from setuptools import setup

setup(
    name='determiners',
    version='0.0.2',
    author='Christopher Brown',
    author_email='chrisbrown@utexas.edu',
    packages=['det'],
    include_package_data=False,
    zip_safe=True,
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'nltk',
        'termcolor',
    ],
    entry_points={
        'console_scripts': [
            'detlearner = det.learner:main',
        ],
    },
)
