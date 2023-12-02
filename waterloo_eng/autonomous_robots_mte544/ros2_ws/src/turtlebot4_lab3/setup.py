from setuptools import setup

package_name = 'turtlebot4_lab3'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'NavAndControlServer = turtlebot4_lab3.NavAndControlServer:main',
            'NavAndControlClient = turtlebot4_lab3.NavAndControlClient:main',
        ],
    },
)
