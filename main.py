# main.py
#   miiii main file
# by: Noah Syrkis

# imports
from src import args_fn


def main():
    print("ji")


if __name__ == "__main__":
    args = args_fn()
    if args.script == "main":
        main()
