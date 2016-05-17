from setuptools import setup

setup(name='xgbmagic',
      version='0.0.2.4',
      description='Data preprocessing and analysis using XGBoost',
      url='http://github.com/mirri66/xgbmagic',
      author='Grace Tang',
      author_email='tsmgrace@gmail.com',
      license='MIT',
      packages=['xgbmagic'],
      install_requires=[
          'pandas',
          'numpy',
          'sklearn',
          'seaborn',
          'unicodecsv'
      ],
      zip_safe=False)
