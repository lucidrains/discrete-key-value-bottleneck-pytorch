from setuptools import setup, find_packages

setup(
  name = 'discrete-key-value-bottleneck-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.4',
  license='MIT',
  description = 'Discrete Key / Value Bottleneck - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/discrete-key-value-bottleneck-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'quantization',
    'memory',
    'transfer learning'
  ],
  install_requires=[
    'einops>=0.4',
    'vector-quantize-pytorch',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
