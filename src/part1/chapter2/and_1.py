def and_gate(x1, x2):
    # y = 0 (w1x1 + w2x2 <= theta)
    # y = 1 (w1x1 + w2x2 > theta)
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2

    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


if __name__ == "__main__":
    print(f"AND(0,0)= {and_gate(0,0)}")
    print(f"AND(1,0)= {and_gate(1,0)}")
    print(f"AND(0,1)= {and_gate(0,1)}")
    print(f"AND(1,1)= {and_gate(1,1)}")
