from fractions import Fraction

def arithmetic_decode(encoded, freq_table):
    low = 0
    high = 1
    decoded = []
    
    while True:
        total_range = high - low
        symbol_range = encoded - low
        total_freq = sum(freq_table.values())
        
        # Find the symbol based on the encoded value
        cum_prob = 0
        for symbol, freq in freq_table.items():
            symbol_prob = Fraction(freq, total_freq)
            symbol_range_in_range = total_range * cum_prob
            if symbol_range_in_range <= symbol_range < symbol_range_in_range + total_range * symbol_prob:
                decoded.append(symbol)
                low = low + symbol_range_in_range
                high = low + total_range * symbol_prob
                break
            cum_prob += symbol_prob
        
        # If we decoded the end symbol, break the loop
        if symbol == '!':
            break
    
    return ''.join(decoded)

# Example usage
encoded = Fraction(32256, 100000)  # Encoded value
freq_table = {'a': 2, 'e': 3, 'i': 1, 'o': 2, 'u': 1,'!':1}  # Symbol frequencies

decoded = arithmetic_decode(encoded, freq_table)
print("Decoded message:", decoded)
