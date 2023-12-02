from setuptools import setup

package_name = 'turtlebot4_lab1'

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
    maintainer='urbanpistek',
    maintainer_email='urbanpistek@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_circle = turtlebot4_lab1.move_circle:main',
            'move_line = turtlebot4_lab1.move_line:main',
        ],
    },
)
