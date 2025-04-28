
# Setup Conda Env
conda env = BSE_env
---
use `source activate BSE_env` to activate
to install additional packages to this env:
conda install -n BSE_env [package]
source deactivate
conda remove -n BSE_env -all
---

# Notes / things to think about / ideas

Inside `_avg_balance.csv` the data is appended to each row in blocks of 4 categories:
(TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType)
following the core values from the limit order book: (chart,time, bid, ask)

(This info was found on line 1838 of the `BSE.py` file)

These values can be paired up and analysed over multiple trials, multiple configurations, and other possibilities;
think t-test like Cliff was talking about today.

After running the market session over 10 sessions with 40 traders, it does take some time. Furthermore statistical analysis of the configurations will take more time, therefore it is important to start building my statistical analysis software early so I can get my results asap after the courswork is released. This can be done prior to the coursework becuase I believe he will make us compare 2 trading algorithms (traders on BSE) possible get us to modify / create a new one and then back test it using statistical analysis. 


# Understanding BSE:

So, the simple robot traders in BSE can all be thought of
as computerized sales-traders: they take customer limit
orders, and do their best to execute the order in the
market at a price that is better than the limit-price
provided by the customer (this is the price below which
they should not sell, or above which they should not buy).
Customer orders are issued to traders in BSE and then the
traders issue their own quotes as bids or asks into the
market, trying to get a better price than the limit-price
specified by the customer. 

## Robot Traders;

GVWY:
- Dumb robot which issues it's quote price at the limit price. 
- Maximises it's chances of finding another trader. 
- Makes 0 Profit. 
- Does not use LOB data

ZIC:
- Read Gode & Sunder's ZIC traders to understand how it works.
- Similar to a human trader

SHVR:
- Uses LOB data.
- For sell orders, it undercuts the best ask on the LOB (i.e. 0.01 less). As long as it is not below the limit price.
- Buy orders SHVR quotes a bid-price that is one penny more than the current buy orders.

SNPR:
- Lurks in the background, waits for the market to near close then steals the deal.
- Rapidly increases the amount it shaves off the best price as time runs out.

ZIP:
- A ZIP trader uses simple machine learning and a shallow heuristic decision tree to dynamically alter the margin that it aims to achieve on the order it is currently working. 

AA:
- Aggressiveness variable added to ZIP which determines how quickly the trader alters its margin. Vytelingum (2006)


Using BSE:

- An individual trader is chosen at random to issue its
current response by invoking the trader’s
getorder() method: this will either return the
value None, signalling that the trader is not issuing
a quote at the current time, or it will return a quote,
i.e. a fresh order to be added to the LOB; if that is
the case then BSE processes the order via a method
called process_order().

- Prices can be constant, or generated according to a deterministic function of time, or
generated at random from a stochastic function: **conditionally heteroscedastic price-generating functions can easily be constructed. **


# Offset functions (page 28->29 of BSEGuide)

In the range variable for the supply and demand schedules, offset functions can be included to add dynamic scheduling. 
Remember: The midpoint between supply and demand schedules is the equilibrium price.

“Note that, in real financial markets, the dynamic variations in prices are often modeled mathematically as a "drunkard's walk", the kind of time series that you'd get if, at each time-step, you add a small random value to whatever the price was on the previous time-step. Approximations to this kind of time-series can be made in BSE by generating random values in the offset function: for example, in Snippet 4.3, random values from specific distributions could be added to gradient, amplitude and wavelength on each call.”


# Done So Far
Inside `tools.py` I have add in some tools:
- Class: `run_market`. Runs market once or for a number of sessions.
- Function: `manage_files`. Manages the outputs of market sessions, puts them into specific directories.
- Class: `getData`. Retrieves important information from the market_session outputs.
Todo:
- Update `getData` to actually work (wont be hard). Just need to figure out which metrics from each file are important and worth analysing.
    - To figure this out, read papers which have used BSE, they should have some testing done.



# Statistics which matter


