import nand_gate


def or_by_nand(x1, x2):
    signal1 = nand_gate.nand_gate(x1, x1)
    signal2 = nand_gate.nand_gate(x2, x2)

    return nand_gate.nand_gate(signal1, signal2)


def and_by_nand(x1, x2):
    signal = nand_gate.nand_gate(x1, x2)

    return nand_gate.nand_gate(signal, signal)


def not_by_nand(x):
    return nand_gate.nand_gate(x, x)


if __name__ == "__main__":
    print(f"OR(0,0)= {or_by_nand(0,0)}")
    print(f"OR(1,0)= {or_by_nand(1,0)}")
    print(f"OR(0,1)= {or_by_nand(0,1)}")
    print(f"OR(1,1)= {or_by_nand(1,1)}")
    print("-----------------------------")
    print(f"AND(0,0)= {and_by_nand(0,0)}")
    print(f"AND(1,0)= {and_by_nand(1,0)}")
    print(f"AND(0,1)= {and_by_nand(0,1)}")
    print(f"AND(1,1)= {and_by_nand(1,1)}")
    print("-----------------------------")
    print(f"NOT(0)= {not_by_nand(0)}")
    print(f"NOT(1)= {not_by_nand(1)}")
