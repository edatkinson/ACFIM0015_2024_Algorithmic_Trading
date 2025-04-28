

# Overview of the type of statistical tests we must do in order to rigorously prove that PT2 is better than PT1

- - -

```
for dir in BTC-USD-*; do
  if find "$dir" -type f -empty | grep -q .; then
    echo "Contains empty file(s): $dir"
  fi
done
```

## Market Scenarios:

For each of these, run them 500 times for 1day data and 100 times for 7 (remember to divide by 7 at the end) day data.

Data: 

We need to partition the data into separate days. I don't know why, but BSE doesn't work otherwise. 

  - BTC-USD: 500 trials per file
    - 10-02-2025 - 1D-5m
    - 11-02-2025 - 1D-5m (done name=BTC-USD-20250211 population = 10-5-5-5 (ZIP,ZIC,SHAVR,GVWY))
    - 31-03-2025 - 06-04-2025 7D-5m : Very volatile. 
    - 03-04-2025 - 09-04-2025 7D-5m : Shoots Up, holds, sharp drop and holds, reverts back
    - 09-04-2025 - 13-04-2025 7D-5m : Very volatilve. 
    - 13/14-04-2025 - 1d-1m-5m : Trending down & high volatility

- - - 
1.
Initially evalute the performance of the bots with this configuration: 20:20:20:20 (so 10 each)
- > Compare performance across data files.
- > select the best & worst performing datafile for further analysis

- - -
2.
Vary the population of traders for best & worst datafile performances:
- > Do all permuations of 10-5-5-5 other traders.
- > Compare the mean, mode, median, std etc

Done:
- BTC-USD-20250211: 10-5-5-5 (ZIP,ZIC,SHAVR,GVWY) 

TODO:
Permutations of 2:1:1:1 (50 total); 200 trials per run
- BTC-USD-20250211-51055
- BTC-USD-20250211-55105
- BTC-USD-20250211-55510

Bigger gap: 3:1:1:1 (84 total); 200 trials per run
- BTC-USD-20250211-21777 
- BTC-USD-20250211-72177 
- BTC-USD-20250211-77217 
- BTC-USD-20250211-77721

Coupled traders: (60 total); 200 trials per run
- BTC-USD-20250211-101055 (done)
- BTC-USD-20250211-105510 (done)
- BTC-USD-20250211-510510 (done)
- BTC-USD-20250211-551010 (done)
- BTC-USD-20250211-105105 (failed)


- - -
3.
Supply & Demand variation: 200 trials per run
"We evaluated PT2 and PT1 under asymmetric market conditions by varying the price elasticity of supply and demand. For example, we created a market with highly elastic demand and inelastic supply by setting a wide price range for demand (70–130) and a narrow range for supply (95–105). This allowed us to test how each trader adapted to a market where buyers were price-sensitive and sellers were not."
- High S, Low D - ds_lh 
- Low S, High D - ds_hl

- - -
4.



