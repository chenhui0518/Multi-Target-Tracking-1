from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d
from ipywidgets import interact
from bokeh.io import push_notebook, show, output_notebook
output_notebook()
from particle_filter_prototype import *

# run particle filter
robot_history, particles_history = run()


def store_positions(particles):
  xs, ys, hs = [], [], []
  for particle in particles:
    x, y, h = particle.get_current_position()
    xs.append(x)
    ys.append(y)
    hs.append(h - pi /2)

  return xs, ys, hs

landmark_xs, landmark_ys = [], []

for landmark in LANDMARKS:
  landmark_xs.append(landmark.x)
  landmark_ys.append(landmark.y)

landmark_source = ColumnDataSource(data = {
    'x' : landmark_xs,
    'y' : landmark_ys
})

x, y, h = robot_history[0].get_current_position()
robot_source = ColumnDataSource(data = {
    'x' : [x],
    'y' : [y],
    'h' : [h - pi / 2]
})

xs, ys, hs = store_positions(particles_history[0])
particles_source = ColumnDataSource(data = {
    'x' : xs,
    'y' : ys,
    'h' : hs
})

p = figure(plot_width = 1000, plot_height = 700)

plot_particles = p.triangle(
  'x', 'y', size = 10,
  fill_color = "violet",
  line_color = "violet",
  fill_alpha = 0.10,
  line_width = 1,
  angle = 'h',
  legend = "particles",
  source = particles_source)

p.square(
  'x', 'y', size = 50,
  fill_color = "orange",
  line_color = "grey",
  fill_alpha = 0.5,
  line_width = 1,
  legend = "landmarks",
  source = landmark_source)

plot_robot = p.triangle(
  'x', 'y', size = 30,
  fill_color = "lightgreen",
  line_color = "olive",
  fill_alpha = 0.75,
  line_width = 1,
  angle = 'h',
  legend = "robot",
  source = robot_source)

p.x_range = Range1d(0, WORLD_SIZE)
p.y_range = Range1d(0, WORLD_SIZE)

def update(w):

  step = (w - 1) // 2 + 1

  if (w == 0):
    print("(initial)")
  elif (w % 2 == 1):
    print("step:", step, "A (move)")
  else:
    print("step:", step, "B (resample) error:", evaluate(robot_history[w], particles_history[w]))

  x, y, h = robot_history[w].get_current_position()
  xs, ys, hs = store_positions(particles_history[w])

  plot_particles.data_source.data['x'] = xs
  plot_particles.data_source.data['y'] = ys
  plot_particles.data_source.data['h'] = hs

  plot_robot.data_source.data['x'] = [x]
  plot_robot.data_source.data['y'] = [y]
  plot_robot.data_source.data['h'] = [h]

  push_notebook()

show(p, notebook_handle = True)
interact(update, w = (0, 2 * NUMBER_OF_STEPS - 2) )
