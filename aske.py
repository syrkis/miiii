# interst_calculation.py
#     Calculate the amount of money you will have after a certain amount of time
# by: Aske Brunken


# imports
import sys
import argparse  # <- for later


# main
def main():  # TODO: allow for use as library.
    principal = float(sys.argv[1])
    rate = float(sys.argv[2])
    months = int(sys.argv[3])
    # principal = float(input("Enter the principal amount: "))
    # rate = float(input("Enter the rate of interest: "))
    # months = float(input("Enter the number of months: "))
    amount = principal * (1.0 + rate) ** months  # calculation
    print(f"you will have {amount} cashmoney youll have after {months} months")
