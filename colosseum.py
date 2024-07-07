# (c) Mikal Rian 2024
#
import numpy as np

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

		elif prob_dist_type=='biased':
			random_n_players = np.random.default_rng().choice(probability_dist, size=[y, 1], replace=False) # replace=False makes instinctively more sense... It means to pool is depleted when populating the colosseum.
			random_n_players = np.rint(random_n_players).astype(int)  # Round to nearest integer. Problem: 2.00 to 2.49 is 2; 2.5 to 2.99 is 3; 49.00 to 49.49 is 49; rest is 50. So if min=2 and max=50, these two extreme values will be even less likely than normal for some of the wide distribution of low beta.
		colosseum = np.zeros([y, x])
		colosseum[:,0:nmin] = 1

		# Populate all games.
		for i in range(nmin, x):
			colosseum[:,i:i+1][i < random_n_players[:,0:1]] = 1

	return colosseum


# Function for auction rounds.
def draw_next_players_to_make_move(matrix, random_sequence=None):
	m, n = matrix.shape[0], matrix.shape[1]
	rand_overlay = np.zeros([m, n]).astype('int8')  # or uint8
	#rand_overlay[:] = random_sequence  # Places the same sequence over every row. (Repeating the random sequence doesn't matter, because each row is a different, independent game.)
	#zeros_index = np.where(matrix == 0)  # Index of original matrix' zeros.
	#rand_overlay[zeros_index] = 0 # =matrix[zeros_index]  # Now we have a new matrix like the original, except that random numbers replaced the original contents. I.e., we removed no-player cells.
	
	# Create random sequence for each row. Remove numbers where org matrix contains 0s.
	# (Sequence is repeated, but, since each row is an independent game, this is not important.)
	random_sequence1 = np.random.choice(range(1, n+1), size=n, replace=False)#.astype('int64')  # For order or rank
	rand_overlay[:] = random_sequence1
	rand_overlay[matrix == 0] = 0

	# Now let's choose who draws next (select highest number from each row).
	# We get an index for the highest number on each row (remember that some cells where 0'd out above).
	# Then we zero out everything, and bring in only the highest draws. This will be next player.
	max_index = rand_overlay.argmax(axis=1)  # Speed 1B matrix = 6.5sec, maybe less with int8...
	rand_overlay[:] = 0
	rand_overlay[range(0, m), max_index] = random_sequence1[max_index]
	matrix[rand_overlay == 0] = 0
	return matrix


# Take in a matrix with 0 to N values in each row. Return matrix with 0 to 1 values per row, selected randomly. 
def draw_next_players(matrix):
	m, n = matrix.shape
	overlay = np.zeros([m, n]).astype('int8')  # or uint8
	random_sequence = np.random.choice(range(1, n+1), size=n, replace=False)#.astype('int64')  # For order or rank	
	overlay[:] = random_sequence  # Places the same sequence over every row. (Repeating the random sequence doesn't matter, because each row is a different, independent game.)
	#  [0, 1, 1]
	#  [1, 0, 0]
	#  [1, 0, 1]

	# Alt 1				10 = 11.86sec, 	1 = 1.11sec
	new = np.zeros([m, n]).astype('int8')
	overlay[matrix == 0] = 0
	highest_index = np.where(overlay == np.amax(overlay, axis=1, keepdims=True))# & (matrix > 0) )
	new[highest_index] = 1

	# Alt 2				10 = 12.55sec,	1 = 0.77sec
	#overlay[matrix == 0] = 0
	#rest_index = np.where(overlay != np.amax(overlay, axis=1, keepdims=True))
	#overlay[rest_index] = 0

	# Alt 3  			10 = 11.39sec,	1 = 0.91sec			index 10 = 29.60sec,	1 = 2.77sec
	#overlay[matrix == 0] = 0
	#rest_index = np.where((overlay != np.amax(overlay, axis=1, keepdims=True)) & (overlay != 0))
	#rest_index = np.where((overlay != np.amax(overlay, axis=1, keepdims=True)) & (matrix != 0))  # This should replace the above two sentence, but the 'matrix!=0' hasn't been tested.
	#overlay[rest_index] = 0

	return new
	#return rest_index
	#return overlay


# Take in a matrix with 0 to N values in each row. Return matrix with 0 to 1 values per row, selected randomly. 
# Faster.
def random_from_each_row(matrix, m, n, overlay, new=None):
	#overlay = np.broadcast_to(np.random.choice(range(1, n+1), size=n, replace=False).astype('int8'), (m, n))
	overlay[:] = np.random.choice(range(1, n+1), size=(1,n), replace=False).astype('int8')  # NB! Max columns for int8 is 256.
	#print(overlay)#; exit()
	#overlay = np.random.choice(range(1, n+1), size=(1,n), replace=False).astype('int8')  # NB! Max columns for int8 is 256.
	#overlay = overlay.repeat(m, axis=0)
	# [0, 1, 1]
	# [1, 0, 0]
	# [1, 0, 1]]
	#matrix = np.array([[0, 1, 1], [0, 0, 0], [1, 0, 1]])

	overlay[matrix == 0] = 0			# What if one row is all zeros? ...........
	#print(overlay)#; exit()
	#highest_index = np.where(overlay == np.amax(overlay, axis=1, keepdims=True))
	highest = np.amax(overlay, axis=1, keepdims=True)
	#print(highest)#; exit()
	highest_index = np.where((overlay == highest) & (highest != 0))# & (matrix != 0))  # matrix!=0 doesn't give me the expected result
	#print('mx: ', *matrix[highest_index])
	#new[highest_index] = overlay[highest_index]
	#new[highest_index] = 1
	return highest_index  # Returning index is much faster, so long as the index is small (say one coord point per row).
	#return new



###########################################################################################
# Garbage → 
# All this is kept for experimentative reference only; it can be deleted.
###########################################################################################

# Remove current highest bidder from play.
def exclude_highest_bidder(players, bids):
	m = players.shape[0]
	max_index = bids.argmax(axis=1)  # What if there is no highest bidder?
	players[range(m), max_index] = 0 # Does this work always?
	return players#, bids

# Remove current highest bidder from play.
def exclude_highest_bidder_bool(players, bids):
	m = players.shape[0]



	highest = np.max(bids, axis=1, keepdims=True)
	index = np.where(bids == highest)
	print(index); exit()
	max_index = bids.argmax(axis=1)  # What if there is no highest bidder?
	print(players[range(m), max_index]); exit()
	players[range(m), max_index] = 0 # Does this work always?
	return players#, bids

# Remove current highest bidder from play.
def exclude_current_highest_bidder(players, bidIDs):
	m = players.shape[0]
	print(bidIDs)
	players[range(m), bidIDs] = 0
	print(players); exit() ###############################################················
	return players

def update_bids(players, bids, increment):
	#players[1][2] = 0; print(players); exit()
	#index = np.where(players != 0)
	#index = np.count_nonzero(players > 0, keepdims=True)  # Skips rows with 0 values.

	i_p = players.argmax(axis=1, keepdims=True)  # Seems keepdims makes no difference.
	#print(i_p)

	#print(index); exit()
	#print(players[index]); exit()
	#bids[index] += increment  # Defect: Ignores the current highest bid; increases instead the player's bid.
	#bids[:] = 0
	#print(bids); print(index)
	#bids[index] = increment + bids.max(axis=1)  # last used. Problem: if one row has no players, the index and bids don't match.
	bids[range(players.shape[0]), i_p] = increment + bids.max(axis=1)  # Seems keepdims makes no difference.
	#bids[np.any(players > 0, axis=1)] += increment
	#bids += increment
	#bids[bids.max(axis=1) != bids.max(axis=1)] = 0
	#bids[players > 0] = increment + bids.max(axis=1)  # This does the same, so long as there's a player in every round.
	return bids

def get_positions_where_row_not_zero(matrix):
	positions = np.zeros([matrix.shape[0], 1]).reshape(-1) * np.nan
	notallzeros = ~np.all(matrix == 0, axis = 1)#, keepdims=True)
	#positions = (a>0)[notallzeros]  # True/False 2D matrix
	positions[notallzeros] = np.argmax(matrix > 0, axis = 1)[notallzeros]#.reshape(-1)#.T  # Gives positions.
	return positions

def update_bidlist(players, bids, increment):
	#b = bids[0].view()
	#l = bids.shape[0]	
	#b = bids[l]

	bids[np.all(players > 0, axis=1)] += increment

	return bidlist

def increment_bids(players, bidIDs, bidz, increment):
	bidIDs[0] = 33
	bidIDs[1] = 99
	players[1] = 0 # whole row

	m = players.shape[0]
	n = players.shape[1]
	mm= np.arange(m).reshape((m, 1)).astype('float16')

	nz = np.isnan 

	non_zero_index = np.any(players > 0, axis=1)  # 1D as True/False pr. row
	players[:,1:][non_zero_index == False] = -1
	print(players)
	pids = np.where(players > -1)
	#col  = pids[1]
	#bidIDs[:,col] = players[:,col]
	col = pids[1]
	print(col); exit()
	bidIDs[col,:] = players[col,:]
	print(bidIDs); exit()

	playerIDs = np.where((players > 0) & (np.isnan(players) == False))
	#bidIDs[playerIDs[0], playerIDs[1]] = playerIDs[1]

	#players[players==0] = np.nan
	z = np.argmax(players, axis=1)
	
	print(players)
	print(playerIDs)
	print(z)
	exit()

	playerIDs = np.any(players > 0, axis=1)  # 1D as True/False pr. row
	playerIDs[playerIDs == False] == np.nan
	zz_ind = np.where(np.isnan(players) == False)
	print("zz_ind ", zz_ind)
	
	temp = np.zeros([m, 1])
	temp[zero_index] = bidIDs[zero_index]

	playerIDs = np.any(players > 0, axis=1)  # 1D
	print(playerIDs); exit()
	bidIDs[:] = playerIDs
	bidIDs[zero_index] = temp[zero_index]

	temp[zero_index] = bidz[zero_index]	

	print(*players); print(zero_index); print(temp); exit()

	bidIDs


	#rand_overlay[:] = random_sequence1
	#rand_overlay[matrix == 0] = 0
	#player_coords = np.array([[range(m)], [bidIDs]])
	# bidz[bidIDs != playerIDs] = bidz

	# Convert positions to ID.
	#players[1] = 0; 
	players[1] = 0
	print(*players)
	playerIDs = np.any(players > 0, axis=1)  # 1D
	print(playerIDs)#; exit()
	mm[playerIDs == False] = np.nan
	print(mm)

	exit()

	#playerIDs = np.any(players > 0, axis=1)#.astype('float16').reshape(m, 1)  # Returns True/False or 1/0
	playerIDs = np.where(players > 0)[1].reshape(m, 1)
	playerIDs_m = np.where(players > 0)[0].reshape(m, 1)

	#bidIDs[]

	print(*playerIDs_m, *playerIDs)
	exit()
	print(*bidIDs)#; exit()

	bidIDs[playerIDs_m,0:1] = playerIDs
	#bidIDs[playerIDs == bidIDs] #= np.nan

	print(*bidIDs); exit()




	index = np.where(np.any(players > 0, axis=1))  # 1D
	print(index); exit()
	print(playerIDs); exit()
	player_list[player_list == False] = np.nan
	print(player_list)
	index = np.where(player_list >= 0)
	#print(index)

	bidIDs[1] = 2
	print("bidIDs ", bidIDs)
	#bidIDs[bidIDs == 0] = np.nan
	bidIDs[index] = player_list[index]
	#bidIDs[:] = player_list

	#bidIDs[player_list == False] = 22
	print(bidIDs); exit()


	players[players == 0] = np.nan
	index = np.where(players > 0)#[1]#.reshape(m, 1)
	#print('index ', index)
	bidIDs[index] = index[1]
	print(players); exit()

	#player_list[np.any(player_list == 0, axis=1)] = -5
	print(player_list); exit()
	#bidIDs[mm, playerlist] = 

	e[np.any(a > 0, axis=1)] = np.nan

	players[1] = False
	player_list = np.any(players > 0, axis=1)
	
	#bidIDs[player_list == True] = -1 # np.nan #players[player_list]
	
	print(player_list)
	print(bidIDs)
	#playerIDs = players[player_list, nn]
	playerIDs = (player_list).nonzero()
	#print(playerIDs); 
	exit()
	#print(bidIDs); print(player_list); exit()
	#bidIDs[:] = range(n)
	bidIDs[player_list] = 9
	print(bidIDs); exit()
	print(players[player_list==True]); exit()
	bidIDs[:] = players[player_list==True]
	bidz[flat_index] += increment
	print(bidIDs); print(bidz); exit()
	return bidIDs, bidz

def shuffle(matrix, random_sequence):
	matrix[:, random_sequence] = matrix
	return matrix


for i in range(0):

	random_sequence1 = np.random.choice(range(1, 4), size=3, replace=False)#.astype('int64')  # For order or rank
	print(*bids, *players,'start →')
	players = draw_next_players_to_make_move(players, random_sequence1)
	bids[players > 0] = bid_increment + bids.max(axis=1)
	print(random_sequence1)
	print(*bids, *players, '\n')

	random_sequence1 = np.random.choice(range(1, 4), size=3, replace=False)#.astype('int64')  # For order or rank
	print(random_sequence1)#; exit()
	players = draw_next_players_to_make_move(players, random_sequence1)
	bid_series[players > 0] = bid_increment + bid_series.max(axis=1)
	print(*bid_series, *players, '\n')

	random_sequence1 = np.random.choice(range(1, 4), size=3, replace=False)#.astype('int64')  # For order or rank
	print(random_sequence1)#; exit()
	players = draw_next_players_to_make_move(players, random_sequence1)
	bid_series[players > 0] = bid_increment + bid_series.max(axis=1)
	print(*bid_series, *players, '\n')

	exit()

	matrix_temp[range(m), bid_id] = bid_series

	print(matrix_temp); exit()

	print(players)








	exit()
	players = draw_next_players_to_make_move(colosseum, random_sequence)
	#print(colosseum); 
	print(players); exit()


	#max_ = np.amax(o, axis=1)  # She who draws the highest number bids first.
	#index_max = np.where(o == o.max())
	nn = o.argmax(axis=1)			# Speed 1B = 6.5sec
	print(nn)
	#nn = (col > 0).argmax(axis=1)	# Speed 1B = 6.5sec

	mm = np.arange(0, m)

	o[:] = 0#np.nan
	o[mm, nn] = colosseum[mm, nn]  # Is reshuffling necessay? I can just place overlay/mask and choose next player to move.
	#o = shuffle(colosseum, random_sequence) # print(o)
	#bids = shuffle(bids, random_sequence)
	#players_1_round = shuffle(players_1_round, random_sequence)
	print(o)
	#print(f'choice N:\t\t{time.time()-start}')
	exit()



	start = time.time()
	mx = np.random.choice(n*m, size=[n, m], replace=False)
	#print(f'no replace:\t\t{time.time()-start}')

	mx = np.arange(0, m*n).reshape([m, n])
	start = time.time()
	np.random.default_rng().shuffle(mx, axis=1)
	#print(f'shuffle:\t\t{time.time()-start}')

	mx = np.arange(0, m*n).reshape([m, n])
	start = time.time()
	new_order = np.random.choice(n, size=n)
	mx[:] = mx[:,new_order]  # In-place is 1.8x slower.
	#print(f'rear in-pl:\t\t{time.time()-start}')
	start = time.time()
	new_order = np.random.choice(n, size=n)
	mx = mx[:,new_order]  # In-place is 1.8x slower.
	#print(f'rear copy:\t\t{time.time()-start}')

	print()


def test():
	n_min_players = 2
	n_games, n_max_players = 4, 4
	beta_ = 1

	from beta import beta_distribution
	prob_dist = beta_distribution(n_max_players*.5, lower_bound=n_min_players, upper_bound=n_max_players, rows=1, cols=n_games, beta_=beta_)[0]
	#prob_dist = np.rint(prob_dist).astype(int) # Distorts extreme values.
	#print(prob_dist)
	colosseum = populate([n_games, n_max_players], nmin=n_min_players, prob_dist_type='biased', probability_dist=prob_dist)
	#print(colosseum)
	sums = [colosseum[x].sum() for x in np.arange(0,colosseum.shape[0])]
	sums = np.array([*sums])
	print(colosseum)
	print(sums.min(), sums.max())


def test2():
	# a[c==2] += 1 → 100M: 0.45sec
	# a[idx]  += 1 → 100M: 0.0sec, yes. 100M x 1,000 loops: 0.42sec 
	import time
	#atts = np.array([[0, 1, 1], [1, 1, 1]])
	atts = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
	bids = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
	N = 3
	R = 5
	#atts = np.random.default_rng().integers(2, size=[N, N])
	#boggok = np.zeros([N, N]).astype('float16')
	#boggok[:,1:2] = 2
	#indexx = np.where(boggok==2)
	#atts[:,1:2] = 2
	#print(boggok); print(atts)
	bids = atts.copy()
	#print(atts)

	# Initialise before
	m, n = atts.shape
	overlay = np.zeros([m, n]).astype('int8')
	new = np.zeros([m, n]).astype('int8')
	#random_sequence = np.random.choice(range(1, n+1), size=n, replace=False)#.astype('int64')
	seq = np.random.choice(range(1, n+1), size=n, replace=False).astype('int8')
	seq2 = np.random.choice(range(1, n+1), size=(m, 1), replace=False).astype('int8')

	times = np.zeros([R, 1])
	start = time.time()
	for i in range(R):
		t = time.time()
		#overlay = np.random.choice(range(1, n+1), size=(m, n), replace=True).astype('int8')
		#overlay[:] = seq2							# 100M x 1,000 =  6.5sec
		#overlay = np.broadcast_to(seq, (m, n))		#                 0.015
		#overlay = seq.repeat(n, axis=0)            #  100M x 10    =  4.80sec            
		#overlay = np.tile(seq, (m, 1))				# 100M x 1,000 = 76.68sec

		#atts[boggok == 2] += 1
		#atts[indexx] += 1
		#atts = draw_next_players(atts)
		#atts = random_from_each_row(atts, m, n, overlay, new)
		att_index = random_from_each_row(atts, m, n, overlay, new)
		print(att_index)
		bids[att_index] += 2
		#print(*bids)
		#atts = draw_next_players_to_make_move(atts)
		#atts[draw_next_players(atts)] = 1
		times[i] = time.time()-t
	end = time.time()-start

	#print(atts); exit()
	#bids[att_index] += 10
	#print(att_index)
	#print(atts)
	#print(atts[att_index]+99)
	#print(*bids)

	#print(f'All\t\t{end}\nMin\t\t{times.min()}\n')

	#print(atts)


if __name__ == '__main__':
	test()
	test2()