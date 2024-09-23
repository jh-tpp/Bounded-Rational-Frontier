import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import importlib

# Import your custom modules
import Modules.InformationTheoryFunctions
import Modules.VisualizationFunctions
import Modules.BlahutArimotoAOS
import Modules.ExampleSetup

# # Reload the modules in case they have been updated
importlib.reload(Modules.InformationTheoryFunctions)
importlib.reload(Modules.VisualizationFunctions)
importlib.reload(Modules.BlahutArimotoAOS)
importlib.reload(Modules.ExampleSetup)

# Import functions from those modules for direct use
from Modules.InformationTheoryFunctions import *
from Modules.VisualizationFunctions import *
from Modules.BlahutArimotoAOS import *
from Modules.ExampleSetup import *
