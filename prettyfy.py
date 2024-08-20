
def hrf(num):
    """
    Convert a number to a human-readable format with suffixes to denote scale.
    :param num: The number to format.
    :return: A string representing the number in a more readable format with appropriate suffix.
    """
    if abs(num) < 1.0:
        return "%1.4f" % num
    if abs(num) < 10:
        return "%2.1f" % num
    if abs(num) < 1000:
        return "%4d" % int(num)
    magnitude = 0
    suffixes = ['K', 'M', 'B', 'T']  # Thousands, Millions, Billions, Trillions
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # Format number to one decimal place and append the appropriate suffix.
    return '{:.1f}{}'.format(num, suffixes[magnitude-1])