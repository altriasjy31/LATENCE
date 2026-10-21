"""Explicit migration aliases from the former NGH package.

New code should use NBS names. These aliases keep old experiments readable while
making the BoxSquaredEL integration explicit in the primary API.
"""

from .config import NBSConfig as NGHConfig
from .hierarchy import NBSHierarchyAdapter as NGHHierarchyAdapter
from .matcher import NBSGatedDeltaAttnRes as NGHGatedDeltaAttnRes
from .model import HomogeneousNBSModel as HomogeneousNGHModel
from .model import ProteinGONBSModel as ProteinGONGHModel
from .types import NBSMatchOutput as NGHMatchOutput
from .types import NBSNeighborhoodHierarchy as NGHNeighborhoodHierarchy
from .types import NBSQueryCondition as NGHQueryCondition
