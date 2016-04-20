from setuptools import setup

setup(name='xgbmagic',
      version='0.1',
      description='Data preprocessing and analysis using XGBoost',
      url='http://github.com/mirri66/xgbmagic',
      author='Grace Tang',
      author_email='tsmgrace@gmail.com',
      license='MIT',
      packages=['xgbmagic'],
      install_requires=[
          'xgboost',
          'seaborn',
          'pandas',
          'numpy',
          'sklearn'
      ],
      zip_safe=False)
