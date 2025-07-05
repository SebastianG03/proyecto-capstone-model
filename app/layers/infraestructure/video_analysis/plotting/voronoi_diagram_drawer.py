import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mplsoccer import Pitch

home_team_coordinates_x = []
home_team_coordinates_y = []
home_team_coordinates = []

rival_team_coordinates_x = []
rival_team_coordinates_y = []
rival_team_coordinates = []

fig, ax = plt.subplots(figsize=(13, 8.5))
pitch = Pitch(
    pitch_type='statsbomb', pitch_color='black', line_color='white', 
    positional=True, half=False, axis=True, label=True, tick=True,)

pitch.draw(ax=ax)

