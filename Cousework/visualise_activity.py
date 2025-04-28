import pandas as pd
import matplotlib.pyplot as plt
import argparse


def visualise_trades(blotter_path, tape_path, trader_id="P00"):
    # Load the CSV file without a header and assign column names.
    df = pd.read_csv(blotter_path, header=None,
                     names=["Trader", "Type", "Time", "Price", "Counterparty", "OtherTrader", "Quantity"])

    # Load the tape price data
    tape_df = pd.read_csv(tape_path, header=None,
                          names=["Event Type", "Time", "Price"])

    # Convert the 'Time' column to numeric seconds
    tape_df["Time"] = pd.to_numeric(tape_df["Time"], errors="coerce")
    tape_df["Price"] = pd.to_numeric(tape_df["Price"], errors="coerce")
    tape_df.dropna(subset=["Time", "Price"], inplace=True)

    # Ensure Time and Price are numeric.
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Fill missing values in trader columns.
    df["Trader"] = df["Trader"].fillna("").astype(str)
    df["Counterparty"] = df["Counterparty"].fillna("").astype(str)
    df["OtherTrader"] = df["OtherTrader"].fillna("").astype(str)

    # Filter only "Trade" events
    trades = df[df["Type"].str.strip() == "Trade"].copy()
    trades.sort_values("Time", inplace=True)

    # Plot market and tape prices
    plt.figure(figsize=(12, 6))
    plt.plot(trades["Time"], trades["Price"], label="Market Price", color="blue", alpha=0.7)
    plt.plot(tape_df["Time"], tape_df["Price"], label="Tape Price", color="orange", alpha=0.7)

    # Filter trades involving the given trader
    trader_mask = ((trades["Trader"].str.strip() == trader_id) |
                   (trades["Counterparty"].str.strip() == trader_id) |
                   (trades["OtherTrader"].str.strip() == trader_id))
    trader_trades = trades[trader_mask].copy()
    trader_trades.sort_values("Time", inplace=True)

    # Assign alternating buy/sell actions
    actions = ["buy" if i % 2 == 0 else "sell" for i in range(len(trader_trades))]
    trader_trades["Action"] = actions

    print(f"Total {trader_id} trades:", len(trader_trades))
    print("Buy trades count:", len(trader_trades[trader_trades["Action"] == "buy"]))
    print("Sell trades count:", len(trader_trades[trader_trades["Action"] == "sell"]))

    # Separate into buy and sell
    buy_trades = trader_trades[trader_trades["Action"] == "buy"]
    sell_trades = trader_trades[trader_trades["Action"] == "sell"]

    # Overlay markers
    plt.scatter(buy_trades["Time"], buy_trades["Price"],
                label=f"Buy ({trader_id})", marker="^", color="green", s=100)
    plt.scatter(sell_trades["Time"], sell_trades["Price"],
                label=f"Sell ({trader_id})", marker="v", color="red", s=100)

    plt.xlabel("Time (s)")
    plt.ylabel("Price")
    plt.title(f"Market Price and {trader_id} Trades")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_profit_over_time(avg_bals_path):
    df = pd.read_csv(avg_bals_path, sep=',', low_memory=False)
    time = df[df.columns[1]]
    pt1 = df[df.columns[9]]
    pt2 = df[df.columns[13]]
    
    plt.step(time, pt1, label='PT1: Balance')
    plt.step(time, pt2, label='PT2: Balance')

    plt.xlabel("Time (s)")
    plt.ylabel("Balance")
    plt.title(f"Balance vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    """
    Example usage
    python3 visualise_activity.py --blotter work/blotters/differentdemand-0/differentdemand-0_0_blotters.csv --tape work/tape/differentdemand-0/differentdemand-0_0_tape.csv --trader P01
    python3 visualise_activity.py --blotter <blotter file> --tape <tape file> --trader P01 

    """
    parser = argparse.ArgumentParser(description="Visualise Market and Trader Activity")
    
    parser.add_argument("--blotter", required=True, help="Path to the blotter CSV file")
    parser.add_argument("--tape", required=True, help="Path to the tape CSV file")
    parser.add_argument("--trader", default="P00", help="Trader ID to highlight (default: P00)")

    parser.add_argument("--profit", required=False)

    args = parser.parse_args()

    visualise_trades(args.blotter, args.tape, args.trader)

    if args.profit is not None:
        plot_profit_over_time(args.profit)
