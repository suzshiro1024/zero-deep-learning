import nand_gate
import and_gate_2
import or_gate


def xor_gate(x1, x2):
    signal1 = nand_gate.nand_gate(x1, x2)
    signal2 = or_gate.or_gate(x1, x2)

    return and_gate_2.and_gate(signal1, signal2)


if __name__ == "__main__":
    print(f"XOR(0,0)= {xor_gate(0,0)}")
    print(f"XOR(1,0)= {xor_gate(1,0)}")
    print(f"XOR(0,1)= {xor_gate(0,1)}")
    print(f"XOR(1,1)= {xor_gate(1,1)}")
