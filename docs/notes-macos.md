# Installation Memo for MacOS

When installing `pyclupan`, the dependency `graphillion` is installed automatically.
On macOS, a dynamic linking error may occur in the `graphillion` shared library due to **duplicate `LC_RPATH` entries**.
This typically happens when `graphillion` is built or installed in a Conda environment, and the same library path is registered multiple times in the binary.
As a result, Python may fail to import `graphillion`, even though the installation itself appears to be successful.

You may fix the `LC_RPATH` entries of the `graphillion` shared object using the following command:
```shell
otool -l ~/.conda/envs/pyclupan-env/lib/python3.13/site-packages/_graphillion.cpython-313-darwin.so | grep LC_RPATH -A 2
```
Remove the duplicated `RPATH` entry using `install_name_tool`:
```shell
install_name_tool -delete_rpath ~/.conda/envs/pyclupan-env/lib ~/.conda/envs/pyclupan-env/lib/python3.13/site-packages/_graphillion.cpython-313-darwin.so
```
The exact path may differ depending on your Conda environment name and Python version.
