# Hybrid Dynamics Modeling for Manipulator and Using RL for Compliance Control
#### [MEU5045] Estimation Theory And Applications (2024Y: Fall Semester) 
<img src="https://github.com/user-attachments/assets/c2d0d1f9-51f4-4131-b816-9b2e8c89ea9a" width="80%" />

</br>

## Install
__Essential 1.__ Clone the repository
```shell
git clone https://github.com/S-CHOI-S/Hybrid-Dynamics-FR3.git hybridynamics
```

__Essential 2.__ Create Conda Environments
```shell
conda env create -f environment.yaml
conda activate hybridynamics
```

__Essential 3.__ Install dependencies
- For the package
    ```shell
    pip install -e .
    ```
- For the control of the manipulator
    - [RBDL](https://github.com/rbdl/rbdl.git): C++ library that contains some essential and efficient rigid body dynamics algorithms.
        > For this project, we use version 3.2.0!
    - [Pybind11](https://github.com/pybind/pybind11.git): A lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code.
    - [GLEW](https://github.com/nigels-com/glew.git): The OpenGL Extension Wrangler Library (GLEW) is a cross-platform open-source C/C++ extension loading library.
    - [GLFW](https://www.glfw.org/): GLFW (Graphics Library Framework) is a lightweight utility library for use with OpenGL.
    ```shell
    sudo apt-get install libglew-dev libglfw3-dev libglfw3
    ```

__Essential 4.__ Build Controller
```shell
mkdir build
cd build
cmake ..
make
```



</br>

### References
- Images

    https://inria.hal.science/hal-02265293/document
    https://franka.de/hs-fs/hubfs/IKR_FR3_noFE_v2-1.jpg?width=560&height=825&name=IKR_FR3_noFE_v2-1.jpg
    https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VHOUViL8dHGfvxCsswPv-Q.png
    
