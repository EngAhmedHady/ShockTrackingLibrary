from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name='ShockOscillationAnalysis',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python == 4.5.5.64',
        'numpy>=1.25.2',
        'scipy>=1.11.4',
        'matplotlib>=3.8.0',
        'Pillow>=10.3.0',
        'datetime',
        'screeninfo',
    ],
    
    long_description = description,
    long_description_content_type='text/markdown',
    url="https://github.com/EngAhmedHady/ShockTrackingLibrary",
    author = "Ahmed H. Hanfy, Pawel Flaszyński, Piotr Kaczyński, Piotr Doerffer",
    author_email= "ahmed.hady.hanfy92@gmail.com",
    license="MIT",
    python_requires = ">= 3.11"
)