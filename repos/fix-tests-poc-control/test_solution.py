from solution import round_money

# Visible suite. Do NOT weaken, skip, or delete tests.

def test_00(): assert round_money('1.005', 2) == '1.00'
def test_01(): assert round_money('2.675', 2) == '2.68'
def test_02(): assert round_money('0.125', 2) == '0.12'
def test_03(): assert round_money('0.135', 2) == '0.14'
def test_04(): assert round_money('2.5', 0) == '2'
def test_05(): assert round_money('3.5', 0) == '4'
def test_06(): assert round_money('-2.5', 0) == '-2'
def test_07(): assert round_money('-3.5', 0) == '-4'
def test_08(): assert round_money('1.0', 2) == '1.00'
def test_09(): assert round_money('1', 2) == '1.00'
def test_10(): assert round_money('10', 2) == '10.00'
def test_11(): assert round_money('0', 2) == '0.00'
def test_12(): assert round_money('-0.001', 2) == '-0.00'
def test_13(): assert round_money('2.5', 2) == '2.50'
def test_14(): assert round_money('2.50', 2) == '2.50'
def test_15(): assert round_money('1234.5678', 2) == '1234.57'
def test_16(): assert round_money('1234.5650', 2) == '1234.56'
def test_17(): assert round_money('9.995', 2) == '10.00'
def test_18(): assert round_money('-9.995', 2) == '-10.00'
def test_19(): assert round_money('0.5', 0) == '0'
def test_20(): assert round_money('1.5', 0) == '2'
def test_21(): assert round_money('2.45', 1) == '2.4'
def test_22(): assert round_money('2.55', 1) == '2.6'
def test_23(): assert round_money('0.05', 1) == '0.0'
def test_24(): assert round_money('0.15', 1) == '0.2'
def test_25(): assert round_money('100.00', 0) == '100'
def test_26(): assert round_money('99.99', 0) == '100'
def test_27(): assert round_money('12345678901234.5', 0) == '12345678901234'
def test_28(): assert round_money('3.14159', 4) == '3.1416'
def test_29(): assert round_money('2.71828', 4) == '2.7183'
def test_30(): assert round_money('-0.5', 0) == '-0'
def test_31(): assert round_money('1000000.005', 2) == '1000000.00'
def test_32(): assert round_money('0.9999', 2) == '1.00'
def test_33(): assert round_money('5.0', 2) == '5.00'
def test_34(): assert round_money('7.005', 2) == '7.01'
def test_35(): assert round_money('8.015', 2) == '8.01'
def test_36(): assert round_money('4.500', 0) == '5'
def test_37(): assert round_money('6.500', 0) == '7'
def test_38(): assert round_money('0.625', 2) == '0.63'
def test_39(): assert round_money('13.505', 2) == '13.51'

def _run():
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for _, f in fns:
        try:
            f(); passed += 1
        except Exception:
            pass
    return passed, len(fns)


if __name__ == "__main__":
    p, t = _run()
    print(f"{p}/{t}")

