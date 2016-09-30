# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:11:22 2016

@author: Marcus Therkildsen
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import calendar
from mpl_toolkits.basemap import Basemap
from matplotlib import colors as c
import googlemaps

gmaps = googlemaps.Client(key=your_key)

"""
Script to plot the mined data from postnord.dk track and trace service.

Plots to make
 - Send, receive and both dates, bar plot
 - Send and receive time distribution
 - Posting time distribution
 - Weight distibution
 - Size distribution
% - Distribution of found T&T numbers, is there a pattern ?
% - most common receiver, sender
 - Map with cities
 - Most used stores
 - ?


data['257059831860059433'].keys()
dict_keys(['shipment_info', 'receiver_info', 'reference_numbers', 'events'])

data['257059831860059433']['shipment_info'].keys()
dict_keys(['delivery_status', 'type_delivery', 'sender', 'weight', 'dimensions', 'type_service'])

data['257059831860059433']['receiver_info'].keys()
dict_keys(['name', 'address'])

data['257059831860059433']['reference_numbers'].keys()
dict_keys(['sender', 'search_colli'])

data['257059831860059433']['events']
-> every time the package has been handled

"""

# Change mathtext to follow text regular size
plt.rcParams['mathtext.default'] = 'regular'

# Change all fonts to certain size unless otherwise stated
font = {'size': 14}
plt.rc('font', **font)


def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time laps in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt==None: dt=datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding-seconds, -dt.microsecond)

# From http://stackoverflow.com/a/7423575/5241172


def autolabel(rects, name):
    # attach some text labels
    for ii, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'% (name[ii]),
                ha='center', va='bottom')


def simpleaxis(ax):
    color = 'black'
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_edgecolor(color)
    ax.tick_params(color=color, labelcolor=color)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_zorder(200)
    ax.spines['bottom'].set_zorder(200)


def y_axis_clean(ax):
    simpleaxis(ax)

    for spine in ['left']:
        ax.spines[spine].set_visible(False)

def factor_plot(the_factor, send_x, send_y, rece_x, rece_y, m, plot_color, lin_width=0.8, c_zorder=1):
    m.plot([send_x, send_x+(rece_x-send_x)*the_factor], [send_y, send_y+(rece_y-send_y)*the_factor], lw=lin_width, color=plot_color, zorder=c_zorder)
    return None

# OPen dataset
with open('post_found_clean.json') as f:
    data = json.load(f)
f.close()


all_tnt = []
for j in data.keys():
    all_tnt.append(j)

tot_len = len(all_tnt)
tot_checked = int(max(all_tnt)) - int(min(all_tnt))

print('Currently ' + str(tot_len) + ' entries in database')
print('We checked ' + str(tot_checked) + ' T&T numbers to get them')
i = int(min(all_tnt))
print('Last found T&T num: ' + str(i))


"""
All, send, receive and both dates, bar plot
"""

s = 0
r = 0
b = 0
id_both = []
for j in data.keys():

    try:
        if 'Elektronisk' in data[j]['events'][0][-1] and 'udleveret' in data[j]['events'][-1][-1]:
            b += 1
            # Check if sunday, then exclude
            if datetime.datetime.strptime(data[j]['events'][-1][0], '%d-%m-%Y %H:%M').weekday() != 6:

                id_both.append(j)
    except Exception:
        pass

    try:
        if 'Elektronisk' in data[j]['events'][0][-1]:
            s += 1
    except Exception:
        pass

    try:
        if 'udleveret' in data[j]['events'][-1][-1]:
            r += 1
    except Exception:
        pass

print("Dates found for")
print("Sender: " + str(s) + ", receiver: " + str(r) + ", both: " + str(b))


send_proper = []
receive_proper = []
send_add_proper = []
rece_add_proper = []
tau = []
bro = []
for j in id_both:

    send_proper.append(datetime.datetime.strptime(data[j]['events'][0][0], '%d-%m-%Y %H:%M'))
    receive_proper.append(datetime.datetime.strptime(data[j]['events'][-1][0], '%d-%m-%Y %H:%M'))

    # Check for Pakkecentrer

    for ev_i in range(len(data[j]['events'])):
        if 'Taulov Pakkecenter - TLP, Danmark' in data[j]['events'][ev_i]:
            tau.append(datetime.datetime.strptime(data[j]['events'][ev_i][0], '%d-%m-%Y %H:%M'))
        if 'Brøndby Pakkecenter - BRC, Danmark' in data[j]['events'][ev_i]:
            bro.append(datetime.datetime.strptime(data[j]['events'][ev_i][0], '%d-%m-%Y %H:%M'))

    if len(tau) < len(send_proper):
        tau.append('')

    if len(bro) < len(send_proper):
        bro.append('')

    send_add_proper.append(data[j]['shipment_info']['sender'])
    rece_add_proper.append(data[j]['receiver_info']['address'])

    # check if any is posted or delivered on a sunday
    if receive_proper[-1].weekday() == 6:
        print(calendar.day_name[receive_proper[-1].weekday()])
        print(j)
        print('')


send_proper = np.array(send_proper)
rece_proper = np.array(receive_proper)
tau = np.array(tau)
bro = np.array(bro)
send_add_proper = np.array(send_add_proper)
rece_add_proper = np.array(rece_add_proper)


"""
http://stackoverflow.com/a/14190143/5241172
days, seconds = duration.days, duration.seconds
hours = days * 24 + seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
"""

proper_len = len(send_proper)

# Time difference
# tdif_arr = rece_proper-send_proper
# need the following instead to account for sundays and seconds
# https://www.pakkelabels.dk/2015/04/21/nu-leverer-post-danmark-ogsaa-pakkepost-om-loerdagen/
posi = 0
tdif_arr_seconds = []
tdif_arr_days = []
for fj in range(proper_len):
    if rece_proper[fj] > send_proper[fj]:
        posi += 1
        sub_day = 0
        for d_ord in range(send_proper[fj].toordinal(), rece_proper[fj].toordinal()):
            d = datetime.date.fromordinal(d_ord)
            if (d.weekday() == 6):
                sub_day += 1
                # print("We passed a Sunday")

        tdif_arr_seconds.append((rece_proper[fj]-send_proper[fj]).total_seconds() - sub_day*24*60*60)

print("posi: " + str(posi))
tdif_arr_seconds = np.array(tdif_arr_seconds)

print("Fastest delivery: " + str(int(round(np.min(tdif_arr_seconds)/(60*60), 0))) + " hours")
print("Slowest delivery: " + str(int(round(np.max(tdif_arr_seconds)/(60*60*24)))) + " days")


"""
>>> from datetime import date
>>> import calendar
>>> my_date = date.today()
>>> calendar.day_name[my_date.weekday()]
'Wednesday'
"""

# Send, rece both bar plot
y_bar = np.array([tot_len, s, r, b, posi])

width = 0.7
ind = np.arange(len(y_bar))

fig, ax = plt.subplots(figsize=(8, 5))
rects = ax.bar(ind, y_bar, width, zorder=100, color='#197319', edgecolor='#197319')
plt.xticks(ind+width/2., ('Total T&T', 'Ship', 'Delivery', 'Both', 'Positive'))
autolabel(rects, y_bar.astype(str))

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_xlim(ind[0]-width, ind[-1]+2*width)

simpleaxis(ax)
plt.tight_layout()
plt.savefig('./plots/send_rece_distri.png',
            dpi=400)
plt.close()

# Receive, send time distribution

s_hours = [send_proper[i].hour for i in range(len(send_proper))]
s_hourCounts, s_hourBins = np.histogram(s_hours, bins=range(25), normed=True)
s_hourCounts = s_hourCounts*100

r_hours = [receive_proper[i].hour for i in range(len(receive_proper))]
r_hourCounts, r_hourBins = np.histogram(r_hours, bins=range(25), normed=True)
r_hourCounts = r_hourCounts*100


'''
Two y axes plot
'''

l_color = '#006400'
r_color = '#be0000'

fig = plt.figure(figsize=(8, 5))
fig.suptitle('Hour shipped/delivered', fontsize=14)
ax = fig.add_subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.bar(s_hourBins[:-1], s_hourCounts, width=1, color=l_color)
ax.set_xticks(range(25))
ax2 = ax.twinx()
ax2.set_xticks(range(25))
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax2.bar(r_hourBins[:-1], r_hourCounts, width=1, color=r_color, alpha=0.8)

# Set labels
ax.set_ylabel('Shipped [%]', color=l_color)
ax2.set_ylabel('Delivered [%]', color=r_color)

# Changing ticks to same amount
ax.axis([0, 24, 0, 1.05 * max(s_hourCounts)])
ax2.axis([0, 24, 0, 1.05 * max(r_hourCounts)])

# ax y ticks
for tl in ax.get_yticklabels():
    tl.set_color(l_color)

# ax2 y ticks
for tl in ax2.get_yticklabels():
    tl.set_color(r_color)

# Remove upper black line
for spine in ['top']:
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

# Remove upper ticks
ax.tick_params(top='off')
ax2.tick_params(top='off')

plt.savefig('./plots/sr_time_distri.png',
            dpi=400, bbox_inches='tight', pad=0)
plt.close()


n_days = int(round(np.max(tdif_arr_seconds)/(60*60*24)))

# tdif seconds
ts_hist = plt.hist(tdif_arr_seconds, 200, normed=True, log=True)

tsCount, tsBins = np.histogram(tdif_arr_seconds, bins=ts_hist[1])
ts_tot = np.sum(tsCount)
tsCount = 100*tsCount/ts_tot

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Shipping time (detailed)', fontsize=14)
rects = ax.bar(tsBins[:-1], tsCount, width=4949, color='#197319', edgecolor='#197319')
ax.set_xticks(24*60*60*np.array(range(int(len(tdif_arr_seconds)/200))))
ax.set_xticklabels(range(n_days+1))
simpleaxis(ax)
ax.set_xlim([0, n_days*24*60*60])
ax.set_xlabel('Days')
ax.set_ylabel('%')
plt.grid(True)
plt.savefig('./plots/tdif_seconds.png',
            dpi=400, bbox_inches='tight', pad=0)
plt.close()

# tdif days

tdCount, tdBins = np.histogram(tdif_arr_seconds, bins=24*60*60*np.arange(n_days+1))
td_tot = np.sum(tdCount)
tdCount = np.cumsum(tdCount)
tdCount = np.around(100*tdCount/td_tot, 1)

# From
# http://www.shocksolution.com/2011/08/removing-an-axis-or-both-axes-from-a-matplotlib-plot/
n_days = 7

fig = plt.figure(facecolor='white', figsize=(8, 5))
ax = plt.axes(frameon=False)
fig.suptitle('Shipping time (daily cumsum)', fontsize=14)
rects = ax.bar(tdBins[:-1][:n_days], tdCount[:n_days], width=24*60*60, color='#197319', edgecolor='k')
ax.set_xticks(24*60*60*np.array(range(n_days+1)))
ax.set_xticklabels(range(n_days+1))
ax.set_yticks([])
ax.set_ylim(0, 110)
simpleaxis(ax)
ax.set_xlim(0, 24*60*60*n_days)
autolabel(rects, tdCount.astype(str))
ax.set_xlabel('Days')
ax.set_ylabel('%')
plt.savefig('./plots/tdif_days.png',
            dpi=400, bbox_inches='tight', pad=0)
plt.close()


print("")

"""
Weight distribution
"""

weight = []
for j in data.keys():
    temp_weight = data[j]['shipment_info']['weight']
    if temp_weight != '':
        weight.append(float(temp_weight.replace(",", ".")))

weight = np.array(weight)

min_weight = np.min(weight)
max_weight = np.max(weight)

wCount, wBins = np.histogram(weight, bins=np.arange(int(np.around(max_weight))))
w_tot = np.sum(wCount)
wCount = 100*wCount/w_tot

print("Weight min: " + str(min_weight) + ", max: " + str(max_weight) + " kg")
print(str(np.around(np.sum(wCount[:6]), 1)) + "% of packages are 5 kg or less")

width = 1
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Weight distribution')
ax.bar(wBins[:-1], wCount, color='#197319', log=True)
simpleaxis(ax)
ax.set_xlim([-width, max_weight])
ax.set_xlabel('Weight [kg]')
ax.set_ylabel('%')
plt.savefig('./plots/weight_distribution.png',
            dpi=400, bbox_inches='tight', pad=0)
plt.close()

"""
Size distribution (meters)
"""

size = []
for j in data.keys():
    temp_size = data[j]['shipment_info']['dimensions']
    if temp_size != '':
        size.append(np.array(temp_size.replace(",", ".").split("x")).astype(float))

# x, y, z in meters
size_arr = np.array(size)

binwidth = 0.02
x_max = np.max(size_arr[:, 0])
xbinBoundaries = np.linspace(0, x_max, 11)
xCount, xBins = np.histogram(size_arr[:, 0], bins=int(np.around(np.max(size_arr[:, 0])/binwidth)))
x_tot = np.sum(xCount)
xCount = 100*xCount/x_tot

y_max = np.max(size_arr[:, 1])
ybinBoundaries = np.linspace(0, y_max, 11)
yCount, yBins = np.histogram(size_arr[:, 1], bins=int(np.around(np.max(size_arr[:, 1])/binwidth)))
y_tot = np.sum(yCount)
yCount = 100*yCount/y_tot

z_max = np.max(size_arr[:, 2])
zbinBoundaries = np.linspace(0, z_max, 11)
zCount, zBins = np.histogram(size_arr[:, 2], bins=int(np.around(np.max(size_arr[:, 2])/binwidth)))
z_tot = np.sum(zCount)
zCount = 100*zCount/z_tot

xyzBins = [xBins, yBins, zBins]

xyzLog = False
normaler = False
com_edge_c = 'k'
c_lw = 0

# Three subplots sharing both x/y axes
f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 5), sharex=True)
f.suptitle('Dimensions')
ax1.bar(xBins[:-1], xCount, width=binwidth, color='#197319', lw=c_lw, edgecolor=com_edge_c, log=xyzLog)
ax2.bar(yBins[:-1], yCount, width=binwidth, color='#be0000', lw=c_lw, edgecolor=com_edge_c, log=xyzLog)
ax3.bar(zBins[:-1], zCount, width=binwidth, color='#006AFF', lw=c_lw, edgecolor=com_edge_c, log=xyzLog)

ax1.set_xlim([0, 0.8])
simpleaxis(ax1)
simpleaxis(ax2)
simpleaxis(ax3)

y_max = np.around(np.max([np.max(xCount), np.max(yCount), np.max(zCount)]))
ax1.set_yticks([0, y_max])
ax2.set_yticks([0, y_max])
ax3.set_yticks([0, y_max])

ax1.set_ylabel('x [%]')
ax2.set_ylabel('y [%]')
ax3.set_ylabel('z [%]')

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0.2)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
ax3.set_xlabel('Size [m]')
plt.savefig('./plots/size_distribution.png',
    dpi=400, bbox_inches='tight', pad=0)
plt.close()

# kg pr. m**3
avg_density = np.mean(weight)/np.mean(np.prod(size, axis=1))

print("Avg. density: " + str(avg_density) + " kg/m^3")
# Pretty close to styrofoam https://en.wikipedia.org/wiki/Density with some
# heavier stuff inside

"""
Distribution of found T&T numbers
"""

all_tnt = []
for j in data.keys():
    all_tnt.append(int(j))
all_tnt = np.array(all_tnt)

all_tnt_sorted = np.sort(all_tnt)

plt.figure()
plt.hist(all_tnt_sorted, 50)
plt.close()

"""
Most common sender and receiver
"""

# Receiver address and name
# data['257059831860049045']['receiver_info']['address']
# data['257059831860049045']['receiver_info']['name']

# Sender address / name
# # data['257059831860049045']['shipment_info']['sender']

rece_arr_name = []
rece_arr_address = []
send_arr_add_name = []
for j in data.keys():
    rece_arr_name.append(data[j]['receiver_info']['name'])
    rece_arr_address.append(data[j]['receiver_info']['address'])
    send_arr_add_name.append(data[j]['shipment_info']['sender'])

rece_arr_name = np.array(rece_arr_name)
rece_arr_address = np.array(rece_arr_address)
send_arr_add_name = np.array(send_arr_add_name)

uniq_rece_name, renam_freq = np.unique(rece_arr_name, return_counts=True)
uniq_rece_add, readd_freq = np.unique(rece_arr_address, return_counts=True)
uniq_send, send_freq = np.unique(send_arr_add_name, return_counts=True)

# get indices of most used
renam_indy = np.argsort(renam_freq)[::-1]
readd_indy = np.argsort(readd_freq)[::-1]
send_indy = np.argsort(send_freq)[::-1]

print('')
print(uniq_rece_name[renam_indy[:10]])
print(renam_freq[renam_indy[:10]])
print('')
print(uniq_rece_add[readd_indy[:10]])
print(readd_freq[readd_indy[:10]])
print('')
print(uniq_send[send_indy[:20]])
print(send_freq[send_indy[:20]])


"""
Most used store
"""

top = 10

# Clean names
clean_uniq = []
for gh in range(top):
    uniq_send[send_indy[gh]] = uniq_send[send_indy[gh]].split(",")[0]

# Normalise
norm_send_freq = 100*send_freq/np.sum(send_freq)

w_all = 0.8

fig, ax = plt.subplots(figsize=(8, 8))
ax.xaxis.grid()
ax.barh(np.arange(top)+0.5, np.around(norm_send_freq[send_indy[:top]][::-1], 1), color='#197319', edgecolor='#197319', zorder=200)
ax.set_title('Top '+str(top)+' shippers')
ax.set_ylim(0, top+w_all)
ax.set_yticks(np.arange(top)+1-(1-w_all)/2)
ax.set_yticklabels(uniq_send[send_indy[:top]][::-1])

simpleaxis(ax)
plt.tight_layout()

ax.set_xlabel('%')

plt.savefig('./plots/most_used_shops.png',
    dpi=400, bbox_inches='tight', pad=0)
plt.close()

"""
Send, receive timeline

send_proper
rece_proper
"""

days_to_subtract = 1

x_s = np.min(send_proper)-datetime.timedelta(days=days_to_subtract)
x_e = np.max(rece_proper)+datetime.timedelta(days=days_to_subtract)

fig, ax = plt.subplots(figsize=(8, 5))
for hj in range(len(send_proper)):
    ax.plot([send_proper[hj], rece_proper[hj]], [10, 0], color='#197319', lw=0.1)
ax.plot([x_s, x_e], [10, 10], color='k', lw=3)
ax.plot([x_s, x_e], [0, 0], color='k', lw=3)
ax.set_xlim(x_s, x_e)
ax.set_ylim(0, 11)
ax.set_yticks([0, 10])
ax.set_yticklabels(['Receive', 'Send'])
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
# Remove top, left and right axis
y_axis_clean(ax)
plt.savefig('./plots/sr_timeline.png',
    dpi=400, bbox_inches='tight', pad=0)
plt.close()


"""
Basemap stuff
"""

# DK. lon-> y, lat-> x
l_lon = 54.5
l_lat = 8

h_lon = 58
h_lat = 13

res = 50
one_ = 0
two_ = 0
c_lw = 1.5
c_ms = 1

lat_s = np.linspace(l_lon, h_lon, res)
lon_s = np.linspace(l_lat, h_lat, res)

lon_, lat_ = np.meshgrid(lon_s, lat_s)
cMap = c.ListedColormap(['g', 'w'])

# Q: l, f, h
m = Basemap(projection='merc',
            resolution='h', llcrnrlon=l_lat, llcrnrlat=l_lon, urcrnrlon=h_lat, urcrnrlat=h_lon)


"""
DK map with most send to cities
Install googlemaps api with
pip install -U googlemaps
from
https://github.com/googlemaps/google-maps-services-python

Basemap installed
https://anaconda.org/anaconda/basemap

On windows python 3.5 is not supported so we turn our attention to Gohlke
http://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap
After download, run pip install some-package.whl
"""

'''
Get coordinates for the shipping addresses
'''
n_to_plot = 15

print(uniq_rece_add[readd_indy[:n_to_plot]])
print(readd_freq[readd_indy[:n_to_plot]])

chosen_cities = uniq_rece_add[readd_indy[:n_to_plot]]
chosen_cities_count = 100*readd_freq[readd_indy[:n_to_plot]]/np.sum(readd_freq)
num_el = len(chosen_cities)
x_coor = np.ones([num_el])*-1
y_coor = np.ones([num_el])*-1


for g in range(len(chosen_cities)):
    try:
        coor_temp = (gmaps.geocode(chosen_cities[g]))[0]['geometry']['location']
        x_coor[g] = coor_temp['lng']
        y_coor[g] = coor_temp['lat']
    except Exception:
        pass

"""
Proper way
https://peak5390.wordpress.com/2012/12/08/matplotlib-basemap-tutorial-plotting-points-on-a-simple-map/
"""

x = np.zeros(num_el)
y = np.zeros(num_el)
for i in range(num_el):
    x[i], y[i] = m(x_coor[i], y_coor[i])


plt.figure(figsize=(8, 5))
m.drawmapboundary(fill_color='w')
m.drawcoastlines(linewidth=0.25)
m.drawcountries()
m.fillcontinents(color='#197319', lake_color='w')
m.scatter(x, y, c=chosen_cities_count, s=40, cmap='viridis', zorder=2000)

m.colorbar()
plt.savefig('./plots/shipping_map.png',
    dpi=400, bbox_inches='tight', pad=0)
plt.close()

###############################################################################

"""
Fancy as fuck map

Steps
- check if send and receive address
- get delivery time
- create plots

The coordinates we will get from a different script so that is taken care of.
My plan is to take the earliest shipment and the latest delivery
"""

first_hour = roundTime(np.min(send_proper) - datetime.timedelta(hours=1), roundTo=60*60)
last_hour = roundTime(np.max(rece_proper) + datetime.timedelta(hours=1), roundTo=60*60)

full_dev_time = rece_proper-send_proper

# The difference
tot_dif = last_hour-first_hour
print(tot_dif)

# around 18 days and 22 minutes.
current_time = first_hour
n_hours = 0
while current_time < last_hour:
    n_hours += 1
    current_time = current_time + datetime.timedelta(hours=1)

# The amount of plots we will need to create for the video
print('We will need ' + str(n_hours) + ' plots')

# Load coordinates database
with open('coor_temp.json') as f:
    coor_data = json.load(f)
f.close()

# Get coordinates of package centers
t_x, t_y = m(coor_data['Taulov Pakkecenter - TLP, Danmark']['coors']['lng'], coor_data['Taulov Pakkecenter - TLP, Danmark']['coors']['lat'])
b_x, b_y = m(coor_data['Broendby Pakkecenter - BRC, Danmark']['coors']['lng'], coor_data['Broendby Pakkecenter - BRC, Danmark']['coors']['lat'])

da_labels = [str(first_hour)[:10]+'\n'+str(first_hour)[11:], str(last_hour)[:10]+'\n'+str(last_hour)[11:]]
lol = last_hour - first_hour
total_days = int(np.ceil((lol.days*60*60*24 + lol.seconds)/(60*60*24)))

timeline_x = []
temp_timeline_x = roundTime(first_hour, roundTo=60*60*24) + datetime.timedelta(days=1)

for i in range(total_days):
    timeline_x.append(temp_timeline_x)
    temp_timeline_x += datetime.timedelta(days=1)

time_line_x_ticklabels = [str(first_hour)[:10]] + [str() for xxx in range(len(timeline_x)-2)] + [str(last_hour)[:10]]


###############################################################################
# Getting timeline

current_time = first_hour
n_hours = 0
tot_stuff = []

send_stuff = []
rece_stuff = []
transit_stuff = []
direct_stuff = []
time_stuff = []

while current_time < last_hour:
    n_hours += 1
    current_time = current_time + datetime.timedelta(hours=1)

    send_this_hour = 0
    rece_this_hour = 0
    direct_this_hour = 0
    transit_this_hour = 0
    stuff = 0

    # Go through all the "posi" entries
    for fj in range(len(send_proper)):
        if send_proper[fj] <= current_time <= rece_proper[fj]:
            stuff += 1
            try:
                if coor_data[send_add_proper[fj]] and coor_data[rece_add_proper[fj]]:
                    # Taking pakkecenter into account
                    if tau[fj] != '' and bro[fj] != '':
                        if tau[fj] < bro[fj]:
                            # Between send
                            if send_proper[fj] <= current_time <= tau[fj]:
                                send_this_hour += 1

                            elif tau[fj] <= current_time <= bro[fj]:
                                transit_this_hour += 1

                            elif bro[fj] <= current_time <= rece_proper[fj]:
                                rece_this_hour += 1

                        # Fra Sjælland til Jylland
                        elif tau[fj] > bro[fj]:
                            # Between send
                            if send_proper[fj] <= current_time <= bro[fj]:
                                send_this_hour += 1

                            elif bro[fj] <= current_time <= tau[fj]:
                                transit_this_hour += 1

                            elif tau[fj] <= current_time <= rece_proper[fj]:
                                rece_this_hour += 1

                    elif tau[fj] != '' and bro[fj] == '':
                        # Between send
                        if send_proper[fj] <= current_time <= tau[fj]:
                            send_this_hour += 1

                        elif tau[fj] <= current_time <= rece_proper[fj]:
                            rece_this_hour += 1

                    elif tau[fj] == '' and bro[fj] != '':
                        # Between send
                        if send_proper[fj] <= current_time <= bro[fj]:
                            send_this_hour += 1

                        elif bro[fj] <= current_time <= rece_proper[fj]:
                            rece_this_hour += 1

                    # Direct
                    elif tau[fj] == '' and bro[fj] == '':
                        direct_this_hour += 1

            except Exception:
                # print('No coor')
                pass

    # Append all the stuff
    send_stuff.append(send_this_hour)
    rece_stuff.append(rece_this_hour)
    direct_stuff.append(direct_this_hour)
    transit_stuff.append(transit_this_hour)

    tot_stuff.append(stuff)
    time_stuff.append(current_time)


send_stuff = np.array(send_stuff)
rece_stuff = np.array(rece_stuff)
direct_stuff = np.array(direct_stuff)
transit_stuff = np.array(transit_stuff)
tot_stuff = np.array(tot_stuff)
time_stuff = np.array(time_stuff)
###############################################################################

skipped_hours = 398

current_time = first_hour + datetime.timedelta(hours=skipped_hours)
n_hours = skipped_hours

send_color = '#1E3B93'
receive_color = '#660000'
transit_pc_color = '#800080'
direct_color = '#CC8400'
transit_lw = 3
transit_zorder = 2500
rece_zorder = 2700
send_zorder = 2000
std_lw = 1
markers_ms = 8
markers_z = 5000
timeline_lw = 4

while current_time < last_hour:
    n_hours += 1
    current_time = current_time + datetime.timedelta(hours=1)

    fig = plt.figure(figsize=(8, 10))
    ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=3, rowspan=5)

    m.drawmapboundary(fill_color='w')
    m.drawcoastlines(linewidth=0.25)
    m.drawcountries()
    m.fillcontinents(color='#197319', lake_color='w')

    print('Searching ' + str(n_hours) + ' hour')
    # Go through all the "posi" entries
    for fj in range(len(send_proper)):
        if send_proper[fj] <= current_time <= rece_proper[fj]:
            try:
                # Find coordinates
                if coor_data[send_add_proper[fj]] and coor_data[rece_add_proper[fj]]:

                    s_x, s_y = m(coor_data[send_add_proper[fj]]['coors']['lng'], coor_data[send_add_proper[fj]]['coors']['lat'])
                    r_x, r_y = m(coor_data[rece_add_proper[fj]]['coors']['lng'], coor_data[rece_add_proper[fj]]['coors']['lat'])

                    # Draw the line between the points
                    # the_factor is between 0 and 1
                    # Taking pakkecenter into account
                    if tau[fj] != '' and bro[fj] != '':
                        if tau[fj] < bro[fj]:

                            """
                            3 different scenarious

                            1. Sender til Taulov
                            2. Taulov til Brøndby
                            3. Brøndby til modtageer
                            """

                            # Between send
                            if send_proper[fj] <= current_time <= tau[fj]:

                                the_factor = (current_time-send_proper[fj])/(tau[fj] - send_proper[fj])
                                factor_plot(the_factor, s_x, s_y, t_x, t_y, m, send_color, std_lw, send_zorder)
                                m.plot(s_x, s_y, 'o', markersize=markers_ms, color=send_color, zorder=markers_z)

                            elif tau[fj] <= current_time <= bro[fj]:

                                the_factor = (current_time-tau[fj])/(bro[fj] - tau[fj])
                                factor_plot(the_factor, t_x, t_y, b_x, b_y, m, transit_pc_color, transit_lw, transit_zorder)

                            elif bro[fj] <= current_time <= rece_proper[fj]:

                                the_factor = (current_time-bro[fj])/(rece_proper[fj] - bro[fj])
                                factor_plot(the_factor, b_x, b_y, r_x, r_y, m, receive_color, std_lw, rece_zorder)
                                m.plot(r_x, r_y, 'o', markersize=markers_ms, color=receive_color, zorder=markers_z)

                        # Fra Sjælland til Jylland
                        elif tau[fj] > bro[fj]:
                            # Between send
                            if send_proper[fj] <= current_time <= bro[fj]:

                                the_factor = (current_time-send_proper[fj])/(bro[fj] - send_proper[fj])
                                factor_plot(the_factor, s_x, s_y, b_x, b_y, m, send_color, std_lw, send_zorder)
                                m.plot(s_x, s_y, 'o', markersize=markers_ms, color=send_color, zorder=markers_z)

                            elif bro[fj] <= current_time <= tau[fj]:

                                the_factor = (current_time-bro[fj])/(tau[fj] - bro[fj])
                                factor_plot(the_factor, b_x, b_y, t_x, t_y, m, transit_pc_color, transit_lw, transit_zorder)

                            elif tau[fj] <= current_time <= rece_proper[fj]:

                                the_factor = (current_time-tau[fj])/(rece_proper[fj] - tau[fj])
                                factor_plot(the_factor, t_x, t_y, r_x, r_y, m, receive_color, std_lw, rece_zorder)
                                m.plot(r_x, r_y, 'o', markersize=markers_ms, color=receive_color, zorder=markers_z)

                    elif tau[fj] != '' and bro[fj] == '':
                        # Between send
                        if send_proper[fj] <= current_time <= tau[fj]:

                            the_factor = (current_time-send_proper[fj])/(tau[fj] - send_proper[fj])
                            factor_plot(the_factor, s_x, s_y, t_x, t_y, m, send_color, std_lw, send_zorder)
                            m.plot(s_x, s_y, 'o', markersize=markers_ms, color=send_color, zorder=markers_z)

                        elif tau[fj] <= current_time <= rece_proper[fj]:

                            the_factor = (current_time-tau[fj])/(rece_proper[fj] - tau[fj])
                            factor_plot(the_factor, t_x, t_y, r_x, r_y, m, receive_color, std_lw, rece_zorder)
                            m.plot(r_x, r_y, 'o', markersize=markers_ms, color=receive_color, zorder=markers_z)

                    elif tau[fj] == '' and bro[fj] != '':
                        # Between send
                        if send_proper[fj] <= current_time <= bro[fj]:

                            the_factor = (current_time-send_proper[fj])/(bro[fj] - send_proper[fj])
                            factor_plot(the_factor, s_x, s_y, b_x, b_y, m, send_color, std_lw, send_zorder)
                            m.plot(s_x, s_y, 'o', markersize=markers_ms, color=send_color, zorder=markers_z)

                        elif bro[fj] <= current_time <= rece_proper[fj]:

                            the_factor = (current_time-bro[fj])/(rece_proper[fj] - bro[fj])
                            factor_plot(the_factor, b_x, b_y, r_x, r_y, m, receive_color, std_lw, rece_zorder)
                            m.plot(r_x, r_y, 'o', markersize=markers_ms, color=receive_color, zorder=markers_z)

                    # Direct
                    elif tau[fj] == '' and bro[fj] == '':

                        the_factor = (current_time-send_proper[fj])/(rece_proper[fj] - send_proper[fj])
                        factor_plot(the_factor, s_x, s_y, r_x, r_y, m, direct_color, std_lw, 100)
                        m.plot(s_x, s_y, 'o', markersize=markers_ms, color=send_color, zorder=100)
                        m.plot(r_x, r_y, 'o', markersize=markers_ms, color=receive_color, zorder=100)

            except Exception:
                # print('No coor')
                pass

    # Plot the massive packet centres
    m.plot(t_x, t_y, 'o', markersize=markers_ms+2, color=transit_pc_color, zorder=markers_z+1000)
    m.plot(b_x, b_y, 'o', markersize=markers_ms+2, color=transit_pc_color, zorder=markers_z+1000)

    # Plot date and number of packages in transit

    # Plot time line
    ax2 = plt.subplot2grid((6, 4), (5, 0), colspan=3)

    ax2.plot(time_stuff[:n_hours], direct_stuff[:n_hours], lw=timeline_lw, color=direct_color)
    ax2.plot(time_stuff[:n_hours], send_stuff[:n_hours], lw=timeline_lw, color=send_color)
    ax2.plot(time_stuff[:n_hours], transit_stuff[:n_hours], lw=timeline_lw, color=transit_pc_color)
    ax2.plot(time_stuff[:n_hours], rece_stuff[:n_hours], lw=timeline_lw, color=receive_color)

    ax2.set_xlim(first_hour, last_hour)
    ax2.set_ylim(1, 3000)

    ax2.set_xticks(timeline_x)
    ax2.set_xticklabels(time_line_x_ticklabels)
    ax2.set_yticks([1, 3000])
    ax2.set_yscale('log')

    ax2.tick_params(axis='x', which='major', pad=13)

    plt.grid(b=True, which='major', color='k', ls='-')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0, top=0.88)

    plt.savefig('./plots/fancy_for_real/'+str(n_hours)+'.png',
        dpi=400, bbox_inches='tight', pad=0)
    plt.close(fig)

tot_stuff_arr = np.array(tot_stuff)

print('Max number of deliveries on a single plot '+str(np.max(tot_stuff_arr)))



"""
openshot on linux to create the video and insert white background

what worked with bludit
<style type="text/css">
video {
        width: 800px;
        height: 600px;
        position: relative;
    }
</style>

<video source src="https://www.mtherkildsen.dk/bl-content/uploads/figures/postnord/postnord_movie.mp4" controls autoplay>
     Your browser does not support the video tag.
</video>
"""
