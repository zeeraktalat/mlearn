from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension




setup(name='Tensor',
      ext_modules=cythonize(
            [Extension
                  ("string2int_CYTHON", 
                  ["string2int_CYTHON.pyx"], 
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=["-stdlib=libc++"], # + anything else you need
                  extra_link_args= ["-stdlib=libc++"] # + anything else you need]
                  )
            ]
      ) 
)
