from mul_layer import mulLayer


if __name__ == "__main__":
    apple_price = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = mulLayer()
    mul_tax_layer = mulLayer()

    subtotal = mul_apple_layer.forward(apple_price, apple_num)
    total_price = mul_tax_layer.forward(subtotal, tax)

    print(f"total_price: {total_price}")

    d_total = 1

    d_subtotal, d_tax = mul_tax_layer.backward(d_total)
    d_price, d_num = mul_apple_layer.backward(d_subtotal)

    print(f"d_subtotal: {d_subtotal}")
    print(f"d_tax: {d_tax}")
    print(f"d_price: {d_price}")
    print(f"d_num: {d_num}")
