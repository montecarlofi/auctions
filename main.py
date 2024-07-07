# (c) Mikal Rian 2023, 2024
#
#
# Simulation to discover how auction rules promotes strategic behaviours and their effect on final price.
# The companion paper "Man of steal" explains the rules and provides justification for the variables and
# their vaules or value ranges.
#
#
# THE SIMULATION
#
# Sequence is as follows:
# A. populate and draw random valuations according to certain probability density functions
# B. initialise first round, then for each round:
# 1. remove those not within own valuation 
# 2. select from attentive players
# 3. place bid
#
#
# COMMENTS
# 
# Instead of programming the players as objects (with behaviours and properties), I've created 
# them as "virtual objects." That is, players are numbers in a matrix. Different numbers correspond to 
# different behaviours and strategies. Doing calculations on a matrix is infinetly faster than running
# operations on individual instances of objects. The results would be the same; the speed very different.
#
# These are the results from doing standard for-loops (required in an OOP scheme):
# The original: 10**4 x 25avg payers, 100 time-steps (bidding rounds) = 100sec, 1,129,000 loops.
# The current version can do the same in 0.4 seconds.
#
# A superfast algorithm could just increase all bids each round instead of having players individually 
# placing bids. (Then filter out by valuation and/or inattention.) Or —
# 
# An even faster algorithm would simply pick the player with the highest valuation in each game and let 
# her win. Simulating the bidding rounds would be altogether unnecessary. To reduce the interactive parts 
# of a complex system, for instance to play around with probability distributions, this would be attractive. 
# Yet it removes the possibilities of measuring the effects from 
# 		· time and timing, 
#		· attention, 
#		· jump bidding (its corresponding psychology and its effectiveness in scaring off players), and
#		· sniping.

# Optimisation possibility: Draw next random bidder from bids matrix; not a player matrix copied from 
# colosseum. A player matrix becomes unnecessary. Update 2024-07-02 — note so self: This is what I did.
# Optimisation possibility II: (Re-) use attention matrix to draw next bidder. This reduces one randomi-
# sation operation from prior solutions. Update 2024-07-03 — note so self: This is what I did.
#
# Optimisation possibility III: Of int8, int16, int32, int64, float16, float32, float64, and bool, the 
# three most rapid are bool, int16, and float32 — the latter two in different order, depending on the 
# task. I've already experimented exptensively to find the current use, but further experimentation 
# and solutions with int16 and bool in the right places could prove fruitful.
#
# By 20240706: 250hrs.
#
# Question:
# To randomise the valuation of an object where it's more likely that it will be undervalued than overvalued, 
# would a beta, gamma, inverse gamma, or a log-logistic distribution be more adequate? Gamma, and many
# others, have no upper bounds, and present problems with respect to forcing bounds. The outliers would 
# have to be re-cast until within bounds. This shoehorning of outliers into a specific range makes an XYZ 
# distribution not an XYZ distribution any longer. Answers at this point: The beta distribution function 
# seems the most fitting for value distribution. I've also used it for population distribution, though 
# a gamma distribution (with open-ended uppers) seems more suitable. The reasons for ruling it out are: (1) 
# that it is for the most part not intereseting to simulate a setting in which, say 200 or 500 people 
# participated in an auction in which 10-40 would otherwise be the norm; (2) the computation involved would 
# grow multifold; and (3) it would no longer be possible to use a matrix for calculations, forcing some 
# kind of nested for-loops solution, further enhancing the computation burden. A beta and gamma distri-
# bution can be quite similar, except that a beta distribtion is forced/normalised into a [0,1) range 
# (wich can be scaled to whatever ranges are needed). But the gamma function produces a fatter long tail,
# i.e., its tail is fatter longer, i.e., it wears of slower than the beta function when both have 
# moderate shapes. That is, of course, because it has many more "soft" outliers (right outside beta's 1 
# bound or, technically, right outside, say, the gamma's 3rd standard deviation). Since a beta distribution 
# can be symmetric, while a gamma function is longer towards one side, a beta function seems appropriate 
# even though, to be strict, there really should be no upper bounds on how many people could or would 
# attend any one, given auction. I am open to suggestions. 
#
#
# TO DO
# • Behavioural attributes
#   timings = np.random.choice(3, [n_games, n_max_players], p=[1/3, 1/3, 1/3])
#   strategies = np.random.choice(strategies, [n_games, n_max_players], p=[1/3, 1/3, 1/3]) # Or use integers for faster matrix operations...?
#   attention as a purposeful time function
#
#
import streamlit as st; st.set_page_config(layout="wide")
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time #as timing
#from itertools import cycle

import colosseum as colosseo  # Library made for this sim.
import display as disp  # Library made for this sim.
#import auction_funcs as a  # Library made for this sim.
from beta import beta_distribution  # Library made for this sim.
#from gamma import gamma_distribution  # Library made for this sim.


# Visual interface and presentation structure
disp.streamlit_hide(st.markdown)
#disp.head2()
#st.subheader('Auction simulation')
#st.markdown("<h3 style='text-align: center; color: darkgrey;'>A u c t i o n &nbsp;&nbsp; s i m u l a t o r</h3>", unsafe_allow_html=True)
#st.markdown("<span style='color: pink; centered: centered; text-alignment: center'>Auction simulator I</span>", unsafe_allow_html=True)
tab_sim, tab_info, tab_quick, tab_comments = st.tabs(['Simulator', 'Info', 'Quick start', 'Technical comments'])
with tab_sim:
	Controls0, Controls1, Plotter, Histogram = st.columns([2, 2, 6, 4])
with tab_info:
	txt = open('info.txt','r').read()
	st.markdown(txt)
with tab_quick:
	txt = open('quick-start.txt','r').read()
	st.markdown(txt)
with tab_comments:
	txt = open('comments.txt','r').read()
	st.markdown(txt)


# User variables (Streamlit)
with Controls0:
	n_games = st.slider('N games', min_value=100, max_value=10**4, value=100, step=100, key=f"key_n_games", disabled=False)
	#n_max_players = st.slider('max players', min_value=2, max_value=50, value=50, step=1, key=f"key_n_max_players", disabled=False)
	#n_min_players = st.slider('min players', min_value=2, max_value=50, value=2, step=1, key=f"key_n_min_players", disabled=True)
	slider = st.select_slider("players", list(range(2, 51)), value=(2, 50))
	n_min_players, n_max_players = slider[0], slider[1]
	mean_valuation = st.slider('mean valuation', min_value=.5, max_value=1.5, value=.8, step=.5, key=f"key_mean_valuation", disabled=True)
	n_snipers = st.slider('snipers', min_value=0, max_value=10, value=0, step=1, key=f"key_n_snipers", disabled=True)
with Controls1:
	beta_valuation = st.slider('β valuation', min_value=1., max_value=32., value=2.5, step=.25, key=f"key_beta_valuation", disabled=False)
	beta_population = st.slider('β population', min_value=1., max_value=16., value=2.25, step=.25, key=f"key_beta_population", disabled=False)
	attention_threshold = st.slider('attention treshold', min_value=0., max_value=1., value=.95, step=.05, key=f"key_attention_threshold", disabled=True)
	time_mode = st.slider('time mode', min_value=0, max_value=1, value=0, step=1, key=f"key_time_mode", disabled=True)
	#time_mode = st.slider("time mode", ['fixed', 'extended'], value='fixed')

# User variables (from userinterface; uncomment these when not using a GUI/Streamlit)
#n_games, n_max_players = 10**4, 50
#n_max_players = 10
#n_min_players = 2
#mean_valuation = 0.8
#beta_population = 2.25
#beta_valuation = 2.5
#attention_threshold = .05 # 19/20 # 0.95


# Sim variables I
attention_integer = 20*attention_threshold+1
upper_bound = 1.05
bid_increment = 1/100
n_time_steps = 100
# A few shortcut names
m, n = n_games, n_max_players
NN = n_time_steps


# Sim variables II — to be implemented
strategies = ['inc', 'jump', 'snipe']
timing = ['start', 'mid', 'endgame']
sensitivity_to_jumps = [0, 1] # [1] is synonymous to saying, "I'll just test the waters, and see if I can get the item for cheap. If not, I'll leave." In other words, jump-sensitivity ≈ low valuation.


# Sim variables III — three matrices that will be used throughout
bids = np.zeros([m, n]).astype('float32')		 # Updated each round
attentions = np.zeros([m, n], order='F').astype('float32')	 # Re-used each round
overlay = np.zeros([m, n]).astype('int8')		 # Re-used each round
#players = np.zeros([m, n]).astype('bool')  	 # Re-used each round. Temp representation of colosseum.


# Get random valuations
valuations = beta_distribution(mean_valuation, 0, upper_bound, n_games, n_max_players, beta_=beta_valuation).astype('float32')  # For flat array: 1, n_games*n_max_players


# Populate the Colosseum
prob_dist = beta_distribution(n_max_players/2, lower_bound=n_min_players, upper_bound=n_max_players, rows=1, cols=n_games*n_max_players, beta_=beta_population).astype('float32')
prob_dist = prob_dist.squeeze(axis=0)  # Reduce by one dimension (to [] from [[]]).
colosseum = colosseo.populate([n_games, n_max_players], nmin=n_min_players, prob_dist_type='biased', probability_dist=prob_dist).astype('float32')  # 'uniform' or 'biased' → 'uniform' is ~5.5x faster
indices = np.where(colosseum == 1)
colosseum[indices] = valuations[indices]  # colosseum[colosseum == 1] = valuations[np.where(colosseum == 1)]


# Clean up and save RAM
indices = None
valuations = valuations[:1000]  # We keep 1000 data points to display the distribution later.
prob_dist = prob_dist[:1000]    # Same.


# Simulation round: 1 (initialisation)
first_valid = np.argmax(colosseum > 0, axis=1, keepdims=True)  # Whoever is the first available player makes the initial bid. (Returns position as column.)
bids[range(m), first_valid.ravel()] = bid_increment            # Place bid. (Positions are flattened to work as index with range.)


# Simulation rouds: 2...N
start_loop = time.time() # times0 = np.zeros([NN])
for i in range(NN):  # There must be an initialised first high bid per row. If not, we'd have to implement many more checks per loop.
	# t = time.time()
	highs = bids.max(axis=1, keepdims=True)
	#high_ix = bids.argmax(axis=1, keepdims=False)  # Flat or tall index?
	#high_ii = bids.argmax(axis=1, keepdims=True)#.unravel_index(2, 1)

	colosseum[colosseum <= highs] = False

	# Re-using attention as a draw-next-player mechanism is fast, but will not be possible once attention ranges are incorporated.
	attentions[:] = np.random.default_rng().integers(0, attention_integer, size=[1, n]) # Here we cast one line over all rows.	 try size=(1, n) instead to see if faster
	attentions[colosseum == 0] = False
	attentions[bids == highs] = False

	# Randomly select next bidder.
	# Best time 10**4 x 50 x 1000: 9.5 sec (vs 16)
	att_index = np.where(np.amax(attentions, axis=1, keepdims=True))
	
	# Best time 10**4 x 50 x 1000: 13 sec (vs 16)
	#current_highs_for_attentives = bids[att_index[0]].max(axis=1, keepdims=False)  # Standing high bids for attentive players (we do not care about high bids for players not bidding in this round).
	#bids[att_index] = current_highs_for_attentives + bid_increment
	
	# Best time 10**4 50 x 1000: 12.5sec (vs 16)
	highs_attentive = highs[att_index[0]]
	bids[att_index] = highs_attentive.T + bid_increment

	#print(*bids, '\t\t', *colosseum, '\t\t\t', att_index, '')	
	#print(*bids, '\t\t\t\t', *attentions, '')	
	#times0[i] = time.time()-t
end_loop = time.time()


# Results
last_line = bids[:n_games:]
winners_vals = np.amax(last_line, axis=1, keepdims=True)
winners_mean = winners_vals.mean()
#winners_pos  = np.argmax(last_line, axis=1, keepdims=True)  # To know which player won.


# Results to console (in text)
msg = f'N = {n_games}.  Mean: {winners_mean}'
below_80 = np.count_nonzero(winners_vals < 0.8) / n_games / 0.01
below_90 = np.count_nonzero(winners_vals < 0.9) / n_games / 0.01
above_100 = np.count_nonzero(winners_vals > 1.00) / n_games / 0.01
above_105 = np.count_nonzero(winners_vals > 1.05) / n_games / 0.01
#print("Below 80%: ", below_80)
#print("Below 90%: ", below_90)
#print("Above 100%: ", above_100)
#print("Above 105%: ", above_105)
#print(msg)
#print(f'\nmean: {avg}')
print(f'loop0: {end_loop-start_loop} seconds\t')
#disp.plot_winning_bids(winners_vals, msg)
avg = np.arange(n_games).astype('float32')
avg[:] = winners_mean


# Results to visual presentation
with Controls0:
	#st.write("Sim time:", round(end_loop-start_loop, 4), 'sec')
	pass

with Plotter:
	#disp.scatter(winners_vals)

	plt.figure(figsize=(7, 6))		#, dpi=300)
	plt.plot(winners_vals, '*')
	plt.plot(avg, '-', color='red')
	y_min = 0.5
	if n_max_players >= 35:
		y_min = 0.7
	elif n_max_players >= 20:
		y_min = 0.6
	plt.yticks(np.arange(y_min, 1.01, .05))
	st.pyplot(plt.gcf())

	st.write("Simulation run time: ", round(end_loop-start_loop, 4), ' seconds')

	# Look at error bars for graphing in an alternative way.

with Histogram:
	disp.histogram(valuations.ravel()[:1000], title='', undertitle='valuation distribution')  # The first 1000 is ok; more makes the visualisation slow.
	disp.histogram(prob_dist[:1000], extent=[2, 50], title='', undertitle='game population distribution')  # The first 1000 is ok; more makes the visualisation slow.

#st.subheader('Auction simulation')