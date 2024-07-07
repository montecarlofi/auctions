import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt

def plot_n_players(colosseum, n_bins=None):
	#sums = [colosseum[x].sum() for x in np.arange(0,colosseum.shape[0])]
	#sums = np.array([*sums])

	sums = np.count_nonzero(colosseum, axis=1)
	msg = f'max: {sums.max()}  min: {sums.min()}  mean: {sums.mean()}'

	x, y = colosseum.shape[1], colosseum.shape[0]
	xy = x * y
	if y >= 10**4:
		bins = 1000
	elif y >= 10**3 and x < 10**4:
		bins = 100
	elif y > 100 and y < 10**3:
		bins = 50
	else:
		bins = y

	if y > x:
		bins = x
		bins = 100 if bins > 100 else bins-1
	else:
		#bins = int(x/10)
		bins = sums.max()-sums.min()

	if y >= 100:
		bins = int(y/10)-1 if y < x else int(x)-1
	else:
		bins = y if y < x else int(x)-1

	bins = y if y < x else x

	if n_bins != None:
		bins = n_bins

	print(f'max: {sums.max()}  min: {sums.min()}')
	#bins = (colosseum.max()-colosseum.min())
	print(f'bins: {bins}')

	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.suptitle("Games and players")

	ax1.plot(sums, 'o')
	ax1.set_ylabel('N players'); ax1.set_xlabel('game #')
	ax1.set_title(msg)

	ax2.hist(sums, bins=bins)
	plt.xlim(2,x) # gives NB/warning for n_players = 2:"UserWarning: Attempting to set identical left == right == 2 results in singular transformations; automatically expanding."
	ax2.set_xlabel('N players')
	ax2.set_ylabel('n games')
	ax2.set_title("Histogram")

	#plt.title(msg)
	plt.show()	
	#plt.plot(sums, 'o'); plt.show()


def plot_winning_bids(bidlist, msg=None):
	y = bidlist
	x = range(len(bidlist))

	average = y.mean()
	#trendline = np.arange()
	trendline = np.array([average for a in x])

	plt.title(msg)
	plt.xlabel("game"); plt.ylabel("winning bid")
	plt.plot(y, 'o')
	plt.plot(trendline)
	plt.show()


#plt.xticks(range(0,n_games, step)) # plt.xticks(range(1,n_games+1))

def scatter(y_values):
	#xx = np.arange(len(y_values))
	yy = y_values.ravel()

	#yy = yy.ravel()
	data = np.array([range(len(y_values)), yy]).T  # Transpose/reshape to x-values being rows.

	chart_data = pd.DataFrame(data=data, columns=['game', 'result'])

	c = (
	   alt.Chart(chart_data)
	   .mark_circle()
	   .encode(
	   		y=alt.Y('result', axis=alt.Axis(tickCount=10, grid=False)),
	   		x=alt.X('game', axis=alt.Axis(tickMinStep=1, grid=False))#tickCount=1))
	   		#color=~'result')
	   	)
	   #.encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
	)

	st.altair_chart(c, use_container_width=True) # theme="streamlit" throws error


def area(y_values): ## UNFINISHED
	#yy = y_values.ravel()
	yy = y_values

	#yy = np.array([4, 5, 6])

	N = len(yy)
	xx = range(N)

	data = np.array([xx, yy]).T  # Transpose/reshape to x-values being rows.

	chart_data = pd.DataFrame(data=data, columns=['N', 'y'])

	c = (
	   alt.Chart(chart_data)
	   .mark_bar(opacity=0.7)
	   .encode(
	   		y = alt.Y('y', axis=alt.Axis(tickCount=10, grid=False)),
	   		x = alt.X('N', axis=alt.Axis(tickMinStep=1, grid=False), bin=True)#tickCount=1))
	   		#color=~'result')
	   	).configure_mark(
	   		color = 'black',
	   		opacity = 0.9
	   	).configure_text(
	   		color = 'pink'
	   	)
	   #.encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
	)

	st.altair_chart(c, use_container_width=False) # theme="streamlit" throws error


def histogram(y_values, x_values=None, title='', undertitle=None, extent=[0.2, 1.1], step=.01):
	yy = y_values

	N = len(yy)
	if x_values == None:
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
	   		opacity = 0.75
	   	).configure_text(
	   		color = 'pink'
	   	)
	)

	st.altair_chart(c, use_container_width=False)