def step(x):
    if x > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    print(f"step(-0.5)= {step(-0.5)}")
    print(f"step(0.0)= {step(0.0)}")
    print(f"step(0.5)= {step(0.5)}")
