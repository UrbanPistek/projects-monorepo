from setuptools import setup

package_name = 'turtlesim_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='churbvm',
    maintainer_email='urban.pistek@gmail.com',
    description='Turtlesim Publisher',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtle_pub = turtlesim_pub.turtle_pub:main',
        ],
    },
)
