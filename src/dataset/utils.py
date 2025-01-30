import random


def split_list(indices, ratios):
    # Calculate the total and each ratio's size in the list
    split_sizes = [round(r * len(indices)) for r in ratios]
    # Split the list according to calculated sizes
    splits = []
    start = 0
    # Shuffle the indices to avoid any bias
    random.shuffle(indices)

    for size in split_sizes:
        splits.append(indices[start : start + size])
        start += size

    return splits
