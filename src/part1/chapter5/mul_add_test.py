from mul_layer import mulLayer
from add_layer import addLayer

if __name__ == "__main__":
    apple_price = 100
    apple_num = 2
    orange_price = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = mulLayer()
    mul_orange_layer = mulLayer()
    subtotal_layer = addLayer()
    total_layer = mulLayer()

    apple_subtotal = mul_apple_layer.forward(apple_price, apple_num)
    orange_subtotal = mul_orange_layer.forward(orange_price, orange_num)
    subtotal = subtotal_layer.forward(apple_subtotal, orange_subtotal)
    total = total_layer.forward(subtotal, tax)

    print(f"total: {total}")

    d_total = 1
    d_subtotal, d_tax = total_layer.backward(d_total)
    d_apple_subtotal, d_orange_subtotal = subtotal_layer.backward(d_subtotal)
    d_apple_price, d_apple_num = mul_apple_layer.backward(d_apple_subtotal)
    d_orange_price, d_orange_num = mul_orange_layer.backward(d_orange_subtotal)

    print(f"d_subtotal: {d_subtotal}")
    print(f"d_tax: {d_tax}")
    print(f"d_apple_subtotal: {d_apple_subtotal}")
    print(f"d_orange_subtotal: {d_orange_subtotal}")
    print(f"d_apple_price: {d_apple_price}")
    print(f"d_orange_price: {d_orange_price}")
    print(f"d_apple_num: {d_apple_num}")
    print(f"d_orange_num: {d_orange_num}")
