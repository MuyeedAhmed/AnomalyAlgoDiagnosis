import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_axes((0.1,0.8,0.8,0.6)) # create an Axes with some room below

X = np.linspace(0,1,1000)
Y = np.cos(X*20)

ax1.plot(X,Y)
ax1.set_xlabel(r"Original x-axis: $X$")



# create second Axes. Note the 0.0 height
ax2 = fig.add_axes((0.1,0.6,0.8,0.0))
ax2.yaxis.set_visible(False) # hide the yaxis

new_tick_locations = np.array([.2, .5, .9])

X = np.linspace(0,1,1000)
Y = np.cos(X*25)
ax1.plot(X,Y)
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels([1, 4, 7])
ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")


# create second Axes. Note the 0.0 height
ax3 = fig.add_axes((0.1,0.1,0.8,0.0))
ax3.yaxis.set_visible(False) # hide the yaxis

new_tick_locations = np.array([.2, .5, .9])

X = np.linspace(0,1,1000)
Y = np.cos(X*40)
ax1.plot(X,Y, color='blue')
ax3.xaxis.label.set_color('blue')
ax3.spines['top'].set_visible(False)
ax3.set_xticks(new_tick_locations)
ax3.set_xticklabels([1, 5, 90])
ax3.set_xlabel("n_estimator")
plt.show()