# PyQt6/PySide6 Installation on Linux

This guide provides instructions and troubleshooting tips for installing PyQt6 or PySide6 on different Linux distributions.

## Ubuntu

### Installation

1.  **Install from Conda-Forge:**
    Use Conda to install Qt and PySide6 from the `conda-forge` channel.
    ```bash
    conda install -c conda-forge qt pyside6
    ```

2.  **Grant GPU Access:**
    Ensure your user has access to the `video` group to interact with the GPU. A reboot is required for this change to take effect.
    ```bash
    sudo usermod -aG video $USER
    sudo reboot
    ```

### Troubleshooting

**Plugin Loading Errors**

If you encounter errors related to loading Qt plugins (e.g., `Could not find the Qt platform plugin "xcb"`), you may need to explicitly set the plugin path.

*   **Set Environment Variables:**
    ```bash
    export QT_PLUGIN_PATH=$CONDA_PREFIX/lib/qt6/plugins
    # If the error persists, you can also try setting the platform-specific path:
    # export QT_QPA_PLATFORM_PLUGIN_PATH=$CONDA_PREFIX/lib/qt6/plugins/platforms
    ```

*   **Set Path Programmatically:**
    Alternatively, set the library path within your Python script before initializing the application.
    ```python
    import os
    from PySide6.QtCore import QCoreApplication

    # Point to the Qt plugins directory within your Conda environment
    plugin_path = os.path.join(os.environ["CONDA_PREFIX"], "lib/qt6/plugins")
    QCoreApplication.setLibraryPaths([plugin_path])
    ```

## Jetson Orin

### Installation and Configuration

The Jetson Orin platform has Qt5 installed by default. The following steps configure the environment to use Qt6 with the correct EGL integration.

1.  **Install from Conda-Forge:**
    Install Qt version 6 and PySide6.
    ```bash
    conda install -c conda-forge qt=6 pyside6
    ```

2.  **Install System Qt6 QPA Plugins:**
    Install the Qt6 QPA (Qt Platform Abstraction) plugins package to get the necessary `libqxcb-egl-integration.so` library.
    ```bash
    sudo apt update
    sudo apt install qt6-qpa-plugins
    ```

3.  **Set Environment Variables:**
    Configure the environment to use the correct EGL integration and library paths before running your application.
    ```bash
    # 1. Tell Qt to use the 'egl' integration
    export QT_XCB_GL_INTEGRATION=egl

    # 2. Point to the system's EGL integration plugin path
    export QT_XCB_GL_INTEGRATION_PATH=/usr/lib/aarch64-linux-gnu/qt6/plugins/xcbglintegrations

    # 3. Prioritize NVIDIA's EGL libraries over Mesa's
    export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra-egl:$LD_LIBRARY_PATH
    ```

### Troubleshooting

**Import Errors**

If you encounter Python import errors like "maximum recursion depth exceeded," it may be due to conflicts with user-site packages. Exclude them by setting the following environment variable:
```bash
export PYTHONNOUSERSITE=1
```