import yaml
import numpy as np

# Your YAML content
yaml_content = """
acc: !!python/object/apply:numpy.core.multiarray.scalar
- !!python/object/apply:numpy.dtype
  args:
  - f8
  - false
  - true
  state: !!python/tuple
  - 3
  - <
  - null
  - null
  - null
  - -1
  - -1
  - 0
- !!binary |
  rkfhehSu5z8=
"""

try:
    # Safely load YAML content
    data = yaml.safe_load(yaml_content)
    
    # Process the data
    numpy_scalar = np.array(data['acc'][1], dtype=data['acc'][0])
    
    print(numpy_scalar)  # Just an example print statement
except Exception as e:
    print(f"An error occurred: {e}")
