from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='controlpy',
      version='0.1.1',
      description='Python control library',
      long_description='Tools for analysing and synthesising controllers in python. Includes LQR solvers, methods for H2 and Hinf, and other potentially useful (state space) methods.',
      url='http://github.com/markwmuller/controlpy',
      download_url = 'https://github.com/markwmuller/controlpy/archive/0.1',
      author='Mark W. Mueller',
      author_email='mwm@mwm.im',
      license='GPL V3',
      packages=['controlpy'],
      zip_safe=False,
      classifiers=[],
      install_requires=['numpy','scipy'],
      keywords='control lqr robust H2 Hinf Hinfinity',
      tests_require=['cvxpy'],
      )
