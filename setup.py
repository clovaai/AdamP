import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='adamp',
    version='0.3.0',
    author='NAVER Corp.',
    description='AdamP optimizer: Slowing Down the Weight Norm Increase in Momentum-based Optimizers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/clovaai/AdamP',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
