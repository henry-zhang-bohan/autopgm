from setuptools import setup

setup(name='autopgm',
      version='0.1',
      description='Automatically learn Bayesian networks from multiple discrete data sources',
      url='https://github.com/ideo-henry/autopgm',
      author='Bohan Zhang',
      author_email='henryhenry.zhang@mail.utoronto.ca',
      license='MIT',
      packages=['autopgm'],
      install_requires=[
          'scipy >= 0.18.1',
          'numpy >= 1.14.0',
          'pandas >= 0.20.3',
          'networkx >= 2.1',
          'matplotlib >= 2.0.2',
          'pgmpy'
      ],
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.5",
          "Intended Audience :: Developers",
          "Operating System :: Unix",
          "Operating System :: POSIX",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: MacOS",
          "Topic :: Scientific/Engineering"
      ],
      zip_safe=False)
