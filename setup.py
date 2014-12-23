from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='controlpy',
      version='0.0.0.dev1',
      description='Python control library',
      url='http://github.com/markwmuller/controlpy',
      author='Mark W. Mueller',
      author_email='mwm@mwm.im',
      license='GPL V3',
      packages=['controlpy'],
      zip_safe=False,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        ],
      keywords='control lqr robust H2 Hinf Hinfinity',
      )
