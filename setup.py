from setuptools import setup, find_packages

setup(
    name='sampler',
    version='0.0',
    packages=find_packages(),
    url='https://github.com/zhengqigao/Sampler/sampler/',
    license='MIT',
    author='Zhengqi Gao',
    author_email='zhengqi@mit.edu',
    description='A python package for various sampling methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "torch",
    ],
    python_requires='>=2.8',
)

