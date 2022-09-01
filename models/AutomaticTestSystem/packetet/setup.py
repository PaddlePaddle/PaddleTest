from setuptools import setup

setup(
    name='paddleocr_test',
    entry_points={
        'console_scripts': [
            'paddleocr_test = test_ocr_acc:main',
        ],
    }
)
