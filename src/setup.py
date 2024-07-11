import sysconfig
from setuptools import setup, Extension
import pybind11
import pybind11.setup_helpers

# Get Python include and library directories
python_include_dir = sysconfig.get_path('include')
python_library_dir = sysconfig.get_config_var('LIBDIR')

# Define the extension module
ext_modules = [
    Extension(
        "EigenForPython",
        ["src/EigenForPython.cpp"],
        include_dirs=[
            pybind11.get_include(),  # Include pybind11 headers
            python_include_dir       # Include Python headers
        ],
        library_dirs=[
            python_library_dir       # Include Python library directories
        ],
        # libraries=["python" + sysconfig.get_python_version().replace(".", "")],  # Link against the Python library
        language="c++",
    ),
]

# Setup the package
setup(
    name="EigenForPython",
    ext_modules=ext_modules,
    cmdclass={"build_ext": pybind11.setup_helpers.build_ext},
)