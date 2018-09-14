from setuptools import find_packages, setup

requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'gpflow>=1.2',
    'tensorflow>=1.4',
    'pytest',
]

setup(name='orth_decoupled_var_gps',
      version='alpha',
      author="Hugh Salimbeni, Ching-An Cheng",
      author_email="hrs13@ic.ac.uk",
      description=("Orthogonally Decopuled Variational Gaussian Processes"),
      license="Apache License 2.0",
      keywords="gaussian processes, variational inference",
      url="https://github.com/hughsalimbeni/orth_decoupled_var_gps",
      python_requires=">=3.5",
      packages=find_packages(include=["odvgp",]),
      install_requires=requirements,
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
