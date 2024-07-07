# (c) Mikal Rian 2023, 2024
#
# Simulation to discover how auction rules promotes strategic behaviours and their effect on final price.
# The companion paper "Man of steal" explains the rules and provides justification for the variables and
# their vaules or value ranges.
#
# Comment: Instead of programming the players as objects (with behaviours and properties), I've created 
# them as "virtual objects." That is, players are numbers in a matrix. Different numbers correspond to 
# different behaviours and strategies. Doing calculations on a matrix is infinetly faster than running
# operations on individual instances of objects. The results would be the same; the speed very different.
#
# The sequence is as follows:
# A populate and draw random valuations according to certain probability density functions
#   and then run the game
# 1 check if within valuation
# 2 select from those attentive
# 3 place bid
#
#To randomise the valuation of an object where it's more likely that it will be undervalued than overvalued, would a gamma, inverse gamma, or a log-logistic distribution be more adequate?
import streamlit as st; st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from gamma import gamma_distribution  # Library made for this sim.
#import auction_funcs as a  # Library made for this sim.
import time #as timing
import altair as alt


# User variables
# Streamlit
Controls0, Controls1, Plotter, Histogram = st.columns([2, 2, 6, 4])
with Controls0:
	n_games = st.slider('N games', min_value=100, max_value=10**4, value=100, step=100, key=f"key_n_games", disabled=False)
	n_max_players = st.slider('max players', min_value=2, max_value=50, value=50, step=1, key=f"key_n_max_players", disabled=False)
	n_min_players = st.slider('min players', min_value=2, max_value=50, value=2, step=1, key=f"key_n_min_players", disabled=True)
	mean_valuation = st.slider('mean valuation', min_value=.5, max_value=1.5, value=.8, step=.5, key=f"key_mean_valuation", disabled=True)
with Controls1:
	beta_valuation = st.slider('β valuation', min_value=1., max_value=32., value=2.5, step=.25, key=f"key_beta_valuation", disabled=False)
	beta_population = st.slider('β population', min_value=1., max_value=16., value=2.25, step=.25, key=f"key_beta_population", disabled=False)
	attention_threshold = st.slider('attention tresh.', min_value=2, max_value=50, value=2, step=1, key=f"key_attention_threshold", disabled=True)
	time_mode = st.slider('time mode', min_value=1, max_value=2, value=1, step=1, key=f"key_time_mode", disabled=True)

# User variables (from userinterface; uncomment these when not using a GUI)
n_games = 10**4
n_max_players = 50
#n_min_players = 2
#mean_valuation = 0.8
#beta_population = 2.25
#beta_valuation = 2.5
#attention_threshold = .05 # 19/20 # 0.95


# Sim variables
strategies = ['inc', 'jump', 'snipe']
timing = ['start', 'mid', 'endgame']
sensitivity_to_jumps = [0, 1] # [1] is synonymous to saying, "I'll just test the waters, and see if I can get the item for cheap. If not, I'll leave." In other words, jump-sensitivity ≈ low valuation.
attention_integer = 20*attention_threshold+1
upper_bound = 1.05
bid_increment = 1/100
NN = 100
n_time_steps = NN

def rescale(value, old_min=0, old_max=1, new_min=0, new_max=100):
    return ((new_max - new_min) / (old_max - old_min)) * (value - old_min) + new_min

def beta_distribution(mean_value=0.5, lower_bound=0, upper_bound=1, rows=1, cols=1, alpha=None, beta_=6):

	def get_beta(a, upper_bound, mean_value):
		return (upper_bound*a - mean_value*a) / mean_value

	def get_alpha(b, upper_bound, mean_value):
		return (mean_value*b) / (upper_bound-mean_value)

	# Define the shape parameters
	#alpha = 22 # calculated alpha value
	#beta_ = get_beta(alpha, upper_bound, mean_value) # calculated beta value

	#print(beta_); exit()
	#beta_ = 6
	if alpha == None:
		alpha = get_alpha(beta_, upper_bound, mean_value)  # For beta=6, alpha=36
		#print(f"Alpha was not declared. Alpha = {alpha}")
	else:
		beta_ = get_beta(alpha, upper_bound, mean_value)
		#print(f'Alpha is {alpha}. This gives beta {beta_}.')

	# Generate random numbers from the Beta distribution and scale them according to upper bound.
	rand_dist = np.random.default_rng().beta(alpha, beta_, size=[rows,cols]) #* upper_bound
	rand_dist = rand_dist if lower_bound == 0 and upper_bound == 1 else rescale(
							rand_dist[:], 
							new_min = lower_bound,			# Unsure if should use lower_bound (min players) or not. Seems to work ok with lower_bound expressed.
							new_max=upper_bound)  

	return rand_dist  # Use this when calling the function from outside.
	#return rand_dist, alpha, beta_  # For testing.

	#from scipy.stats import beta
	#x = np.linspace(0.6, upper_bound, 10**5)
	#a, b = alpha, beta_
	#pdf = beta.pdf(x, a, b, loc=0, scale=upper_bound)
	#plt.plot(x, pdf, label='Beta')
	#plt.legend()
#bid_increment = int(1)


def populate(size, nmin=None, nmean=None, prob_dist_type='uniform', probability_dist=None):  # Size = x or [y, x]
	if np.isscalar(size):
		y = 1
		x = size
	else:
		y = size[0]
		x = size[1]

	# if nmin == 0  # Should use 0 instead of None?
	if nmin == None or nmin >= x:  # Return full colosseum.
		colosseum = np.ones([y, x])
	else:  # Get randomly populated colosseum.
		if prob_dist_type=='uniform':
			random_n_players = np.random.default_rng().integers(nmin, x+1, size=[y, 1])  # One random number for each row (game).
			#print("uniform")
		elif prob_dist_type=='biased':
			random_n_players = np.random.default_rng().choice(probability_dist, size=[y, 1], replace=False) # replace=False makes instinctively more sense... It means to pool is depleted when populating the colosseum.
			#print('biased')
			#random_n_players = np.random.default_rng().choice(prob_dist[0], size=[1, y], replace=False)#[0]
			
			#random_n_players = [probability_dist]
			random_n_players = np.rint(random_n_players).astype(int)  # Round to nearest integer. Problem: 2.00 to 2.49 is 2; 2.5 to 2.99 is 3; 49.00 to 49.49 is 49; rest is 50. So if min=2 and max=50, these two extreme values will be even less likely than normal for some of the wide distribution of low beta.
			#random_n_players = np.ceil(random_n_players).astype(int)  # Floor to nearest integer.  # This distorts extreme values...?
			#print(*random_n_players)
			
			#random_n_players[:] = random_n_players[:] + 99
			#print(f'y: {y}  random_n_players: {random_n_players}')#; exit()
			#print(random_n_players)
		#colosseum = np.random.default_rng().integers(nmin, x, size=[y, x])
		colosseum = np.zeros([y, x])
		colosseum[:,0:nmin] = 1

		#print(random_n_players)
		#print(random_n_players[0:1,:]) ;exit()
		for i in range(nmin, x):
			#print(f'i: {colosseum[:,i:i+1]}')
			#colosseum[:,i:i+1][colosseum[:,i:i+1] < i] = -99
			#print(i)

			colosseum[:,i:i+1][i < random_n_players[:,0:1]] = 1

			#print(f'i: {i}  {colosseum[:,i:i+1]}  and index: {random_n_players[:,0:1]}')
			#colosseum[:,i:i+1][random_n_players[0:1,:] <= i] = 0

		#print(colosseum)


		# Populate all games.
		#colosseum = np.zeros((n_games,n_max_players))
		#colosseum = beta_distribution(n_max_players/2, n_max_players, n_games, n_max_players, beta_=1) # beta_= 1 with mean in the middle gives flat distribution.
		######### colosseum = np.random.default_rng().integers(n_min_players, n_max_players, size=[n_games, n_max_players])
		#firs_col  = colosseum[0]; print(firs_col); exit()
		#colosseum[:,1:] += 1
		#colosseum[:,2:3] = 100
		#print(colosseum)
		#colosseum[colosseum > colosseum[:,0:1]] = 1  # Filter.
		#colosseum[colosseum > colosseum[:,0:1]] = 0  # Filter.
		#colosseum[:,2:][colosseum[:,2:] >= colosseum[:,0:1]] = 0  # Filter from column 2 (3) by first column.
		#colosseum.itemset((1, 2), -99)
		#colosseum[colosseum > 1] = 1  # Filter.
		#exit()
		#colosseum[:,:1] = 1  # Set first col to 1.
		#colosseum[colosseum > 0] = 1  # Set all numbered cells to 1.
		#print(colosseum); exit()
		#print(colosseum[0].sum()); exit()
		#sums = (colosseum[0].sum()); print(sums); exit()
		#print(np.arange(0,n_games)); exit()
	return colosseum


def histogram(y_values, title='', undertitle=None, extent=[0.2, 1.1], step=.01):
	yy = y_values

	N = len(yy)
	xx = range(N)

	data = np.array([
		yy,
		xx
	]).T

	chart_data = pd.DataFrame(data=data, columns=[undertitle, ' '])

	title = alt.TitleParams(title, anchor='middle')

	c = (
	   alt.Chart(chart_data, title=title, width=360, height=233)
	   .mark_bar(
			# opacity=0.7,
			#size = 1
	   )  
	   .encode(
	   		y = alt.Y(' ', axis=alt.Axis(tickCount=0, grid=False), bin=False),
	   		x = alt.X(undertitle, axis=alt.Axis(tickCount=0, tickMinStep=.1, grid=False), bin=alt.Bin(extent=extent, step=step)),
	   	).configure_mark(
	   		color = 'black', # #ff3399 is nice pink  #FF4B4B is streamlit red  # is streamlit red transparent  # #1F77B4 is pyplot blue
	   		opacity = 0.90
	   	).configure_text(
	   		color = 'pink'
	   	)
	)

	st.altair_chart(c, use_container_width=False)




# Random valuations
valuations = beta_distribution(mean_valuation, 0, upper_bound, n_games, n_max_players, beta_=beta_valuation).astype('float32')  # For flat array: 1, n_games*n_max_players
#plt.hist(valuations.ravel()); plt.show(); exit()

# Behavioural attributes
#timings = np.random.choice(3, [n_games, n_max_players], p=[1/3, 1/3, 1/3])
#strategies = np.random.choice(strategies, [n_games, n_max_players], p=[1/3, 1/3, 1/3]) # Or use integers for faster matrix operations...?


# Random population
prob_dist = beta_distribution(n_max_players/2, lower_bound=n_min_players, upper_bound=n_max_players, rows=1, cols=n_games*n_max_players, beta_=beta_population).astype('float32')
prob_dist = prob_dist[0]  # Flat array (that is, [] and not [[]]).
colosseum = populate([n_games, n_max_players], nmin=n_min_players, prob_dist_type='biased', probability_dist=prob_dist).astype('float32')  # 'uniform' or 'biased' → 'uniform' is ~5.5x faster
#prob_dist = None
indices = np.where(colosseum == 1)
colosseum[indices] = valuations[indices]
#valuations = None
#print(np.count_nonzero(prob_dist)); exit()
# n_list_players = np.random.default_rng().integers(n_min_players, n_max_players, size=n_games, endpoint=True).reshape(n_games,1) # Column of N players per game.


# In theory, I can just increase bids for each round INSTEAD of having players bid, then filter out by valuation and/or inattention. 
# This works, so long as there are no snipers. (Or early/late bidder or jump-bidder with corresponding psychology for its effectiveness among contestants.)

# Another optimisation: Draw next random bidder from bids matrix. This way a player matrix becomes unnecessary. 

# The original: 10**4 x 25avg, 100 time-steps = 100sec, 1,129,000 loops.
# This: 10**4 x 25 , 100 time-steps = 1.05sec

m, n = n_games, n_max_players
players = np.zeros([m, n]).astype('bool')  # Temp representation of colosseum.
attentions = np.zeros([m, n]).astype('float32')
bids = np.zeros([m, n]).astype('float32') ### - 1
overlay = np.zeros([m, n]).astype('int8')



#colosseum = np.array([[0, 3, 4], [3, 3, 3]])
# Initialise
#players[:] = colosseum
#players = a.draw_next_players_to_make_move(players)
#bids[players == True] = bid_increment
first_valid = np.argmax(colosseum > 0, axis=1, keepdims=True)  # Whoever is the first available player makes the initial bid. (Returns position as column.)
bids[range(m), first_valid.ravel()] = bid_increment            # Place bid. (Positions are flattened to work as index with range.)

#print(*bids, '\t', *colosseum, '\t', *attentions, '')#; exit()

m, n = n_games, n_max_players



# USE INT TO TEST SPEEED





#bids[0][1] = bid_increment
# Time descreet in the first 90% of game, then continuous?

# Use highest att to draw next player?

# colosseum, bids, players

#NN = 7
# Run sims
#game_over = 1
#colosseum = colosseum.astype('int8')
#bids = bids.astype('int8')

#colosseum[0], colosseum[1] = [1, 2, 3], [4, 5, 6]
#bids[0],      bids[1]      = [1, 0, 0], [0, 1, 0]
#########·······attentions[:] = 1      # For testing
# New?
#bids[1][2] = 0.3
start_loop = time.time()
times0 = np.zeros([NN])

# Simulation runs 2 to N.
# There must be an initialised first high bid per row. If not, we'd have to implement many more checks per loop.
for i in range(NN):
	t = time.time()
	highs = bids.max(axis=1, keepdims=True)
	#high_ix = bids.argmax(axis=1, keepdims=False)  # Flat or tall index?
	#high_ii = bids.argmax(axis=1, keepdims=True)#.unravel_index(2, 1)

	colosseum[colosseum <= highs] = False

	attentions[:] = np.random.default_rng().integers(0, attention_integer, size=[1, n]) # Here we cast one line over all rows.	 try size=(1, n) instead to see if faster
	attentions[colosseum == 0] = False
	attentions[bids == highs] = False

	# Randomly select next bidder.
	att_index = np.where(np.amax(attentions, axis=1, keepdims=True))
	
	# Best time 10**4 x 50 x 1000: 13 sec (vs 16)
	#current_highs_for_attentives = bids[att_index[0]].max(axis=1, keepdims=False)  # Standing high bids for attentive players (we do not care about high bids for players not bidding in this round).
	#bids[att_index] = current_highs_for_attentives + bid_increment
	
	# Best time 10**4 50 x 1000: 12.5sec (vs 16)
	highs_attentive = highs[att_index[0]]
	bids[att_index] = highs_attentive.T + bid_increment

	#print(*bids, '\t\t', *colosseum, '\t\t\t', att_index, '')	
	#print(*bids, '\t\t\t\t', *attentions, '')	
	times0[i] = time.time()-t
end_loop = time.time()
end_time = end_loop-start_loop





# Read results
last_line = bids[:n_games:]
#last_line -= bid_increment
winners_pos  = np.argmax(last_line, axis=1, keepdims=True)
winners_vals = np.amax(last_line, axis=1, keepdims=True)
#print(winners_vals)
winners_mean = winners_vals.mean()
#print(winners_mean)

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
#avg = [winners_mean for x in range(n_games)]

#plt.hist(valuations.ravel()); plt.title(valuations.mean()); plt.show(); exit()
#print(times0.mean())
#print(times0.sum()/NN)
#print(times1.sum()/NN)
#plt.plot(times0, '*'); 
#plt.plot(range(NN), avg); 
#plt.plot(avg1); 


#

#plt.show()
#chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#print(chart_data); exit()

with Plotter:
	plt.figure(figsize=(7, 6))
	plt.plot(winners_vals, '*')
	plt.plot(avg, '-', color='red')
	y_min = 0.5
	if n_max_players >= 35:
		y_min = 0.7
	elif n_max_players >= 20:
		y_min = 0.6
	plt.yticks(np.arange(y_min, 1.01, .05))
	st.pyplot(plt.gcf())
	st.write("Simulation run time: ", round(end_time, 4), ' seconds')
	#disp.scatter(winners_vals)

	#st.write(f'loop0: {end_loop-start_loop} seconds\t')
	#st.write(winners_vals)
	#xx = range(len(winners_vals))
	xx = np.arange(len(winners_vals))
	yy = winners_vals
	#st.plotly_chart({ '1': 2})
	#plt.hist(valuations[0], bins=100)
	#
	#chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", 'c'])
	#st.scatter_chart(chart_data)
	#print(chart_data); exit()
	#st.scatter_chart(chart_data, *, x=None, y=None, x_label=None, y_label=None, color=None, size=None, width=None, height=None, use_container_width=True)
	#df = px.data.iris()  # iris is a pandas DataFrame
	#fig = px.scatter(df, x="sepal_width", y="sepal_length")
	#event = st.plotly_chart(fig, key="iris", on_select="rerun")
	#event

with Histogram:
	#disp.scatter(winners_vals)
	histogram(valuations.ravel()[:1000], title='', undertitle='valuation distribution')  # The first 1000 is ok; more makes the visualisation slow.
	histogram(prob_dist[:1000], extent=[2, 50], title='', undertitle='game population distribution')  # The first 1000 is ok; more makes the visualisation slow.

	#st.write(np.count_nonzero(valuations.ravel()))


	#st.area_chart(data=data, height=height, width=width, use_container_width=False)
	#st.area_chart(data=prob_dist, height=height, width=width, use_container_width=False)
	#disp.area(valuations)
	#st.image('pic2.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
	#st.image('pic1.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
	#chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
	#st.scatter_chart(chart_data)

	#plt.hist(valuations)
	#plt.hist(valuations[0], bins=100)
	#plt.title(f'X={N}\n, beta={beta_valuation}\n\nmin: {prob_dist.min()}  mean: {prob_dist.mean()}')
	#plt.title(f'X={N}\nalpha={alpha}, beta={beta_}\n{msg}\nmin: {rand_dist.min()}  mean: {rand_dist.mean()}')

	#st.pyplot(plt.gcf())
