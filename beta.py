# (c) Mikal Rian 2024
#
import numpy as np


def rescale(value, old_min=0, old_max=1, new_min=0, new_max=100):
    return ((new_max - new_min) / (old_max - old_min)) * (value - old_min) + new_min


def beta_distribution(mean_value=0.5, lower_bound=0, upper_bound=1, rows=1, cols=1, alpha=None, beta_=6):

	def get_beta(a, upper_bound, mean_value):
		return (upper_bound*a - mean_value*a) / mean_value

	def get_alpha(b, upper_bound, mean_value):
		return (mean_value*b) / (upper_bound-mean_value)

	# Define the shape parameters
	if alpha == None:
		alpha = get_alpha(beta_, upper_bound, mean_value)
	else:
		beta_ = get_beta(alpha, upper_bound, mean_value)

	# Generate random numbers from the Beta distribution and scale them according to upper bound.
	rand_dist = np.random.default_rng().beta(alpha, beta_, size=[rows,cols])
	rand_dist = rand_dist if lower_bound == 0 and upper_bound == 1 else rescale(
							rand_dist[:], 
							new_min = lower_bound,  # Unsure if should use lower_bound (min players) or not. Seems to work ok with lower_bound expressed.
							new_max = upper_bound)  

	return rand_dist  # Use this when calling the function from outside.
	#return rand_dist, alpha, beta_  # Use this for local testing.


def test():
	import matplotlib.pyplot as plt
	import time

	mean_value = 80
	upper_bound = 105
	Y = 10**3
	N = upper_bound
	beta_ = 2.25
	#alpha = beta_
	#alpha = beta_distribution().get_alpha(beta_, N, mean_value)
	#rand_dist = beta_distribution(mean_value, upper_bound, rows, cols, alpha=3)
	#rand_dist, alpha, beta_ = beta_distribution(mean_value, upper_bound, rows=Y, cols=N, beta_=1)  # Try 1.1 or 2 to 8 for wide to tight distribution.
	
	# Works:
	rand_dist = beta_distribution(80, lower_bound=2, upper_bound=N, rows=1, cols=Y*N, beta_=beta_)
	#rand_dist = beta_distribution(N/2, lower_bound=2, upper_bound=N, rows=N, cols=Y, beta_=beta_).ravel()  # Also works
	#rand_dist = np.random.default_rng().choice(rand_dist[0], size=[1, Y], replace=False)#[0]  # Seems unnecessary.
		
	# Works:
	#rand_dist = np.random.default_rng().beta(alpha, beta_, size=[1,Y*N]) # This works...
	#rand_dist = rescale(rand_dist[:], new_max=50, new_min=2)             # ... with this.

	print(rand_dist.min())
	print(rand_dist.max())
	print(rand_dist.mean())

	limit = 0.1
	limit_u = .9
	count_over = (rand_dist > limit_u).mean()
	count_under = np.count_nonzero(rand_dist < limit) / N

	msg = f'{count_under} < {limit}, {limit_u} < {count_over}'
	print(msg)
	#print(f'count: {len(rand_dist[0])}')
	print(f'count: {np.count_nonzero(rand_dist)}')
	#exit()

	import matplotlib.pyplot as plt
	plt.hist(rand_dist[0], bins=100)
	#plt.title(f'X={N}\nalpha={alpha}, beta={beta_}\n{msg}\nmin: {rand_dist.min()}  mean: {rand_dist.mean()}')
	#step = int(upper_bound/10)
	#plt.xticks(range(0,upper_bound, 1)) # plt.xticks(range(1,n_games+1))
	#plt.xlim(-.1,1.2)

	#from scipy.stats import beta
	#x = np.linspace(0.6, upper_bound, 10**5)
	#a, b = alpha, beta_
	#pdf = beta.pdf(x, a, b, loc=0, scale=upper_bound)
	#plt.plot(x, pdf, label='Beta')
	#plt.legend()

	plt.show()

if __name__ == '__main__':
	test()