# Hybrid Dynamics FR3
### Hybrid Dynamics Modeling for Franka Research 3 and Using RL for Compliance Control

</br></br>

## Installation
__Essential 1.__ Clone the repository
```shell
$ git clone https://github.com/S-CHOI-S/Hybrid-Dynamics-FR3.git hybridynamics
```

__Essential 2.__ Install Dependencies
> RBDL, GLFW, Pybind11 etc. should be installed!
- [RBDL](https://github.com/rbdl/rbdl.git): C++ library that contains some essential and efficient rigid body dynamics algorithms.
  > For this project, we use version 3.2.0!

- [GLEW](https://github.com/nigels-com/glew.git): The OpenGL Extension Wrangler Library (GLEW) is a cross-platform open-source C/C++ extension loading library.
- [GLFW](https://www.glfw.org/): GLFW (Graphics Library Framework) is a lightweight utility library for use with OpenGL.
  ```shell
  $ sudo apt-get install libglew-dev libglfw3-dev libglfw3
  ```
- [Pybind11](https://github.com/pybind/pybind11.git): A lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code.

__Essential 3.__ Build the project
> Make the **build** directory
```shell
$ mkdir build
```
```shell
$ cd build
```
```
$ cmake ..
```
```
$ make
```

