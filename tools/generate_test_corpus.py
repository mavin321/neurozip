#!/usr/bin/env python3
import random
import string

def random_text(n=5000):
    chars = string.ascii_letters + string.digits + " .,!?()-_=+"
    return "".join(random.choice(chars) for _ in range(n))

def main():
    print(random_text())

if __name__ == "__main__":
    main()
