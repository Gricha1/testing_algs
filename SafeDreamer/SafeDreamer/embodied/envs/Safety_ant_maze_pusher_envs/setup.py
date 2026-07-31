from setuptools import setup, find_packages

setup(
    name="safety_ant_maze_pusher_envs",
    version="0.1.0",
    description="Safety Ant Maze Pusher Environments",
    packages=find_packages(),  # это автоматически найдет все пакеты
    # или явно укажите пакет:
    # packages=['safety_ant_maze_pusher_envs'],
    package_dir={'': '.'},  # ищите пакеты в текущей директории
    #package_data={
    #    'safety_ant_maze_pusher_envs': ['assets/*', '*.py', '*/*.py'],
    #},
    install_requires=[
        # добавьте зависимости если нужно
        #'mujoco_py'
    ],
    python_requires='>=3.6',
)