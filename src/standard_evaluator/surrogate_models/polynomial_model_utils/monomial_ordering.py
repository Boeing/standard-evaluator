def next_grlex(monomial: list, nind: int) -> list:
    """
    Generates the next monomial in the graded lexicographical ordering.

    :param monomial: The current monomial represented as a list of nonnegative integers
    :type monomial: list
    :param nind: The number of independent variables
    :type nind: int

    :return: The next monomial in the ordering represented as a list of nonnegative integers
    :rtype: list
    """

    # Find index of last nonzero

    last_ind = 0
    for ind in range(nind, 0, -1):
        if monomial[ind - 1] > 0:
            last_ind = ind
            break

    # Construct next monomial

    next_monomial = monomial.copy()

    # Move to degree one if all zeros

    if last_ind == 0:
        next_monomial[nind - 1] = 1
        return next_monomial

    # Move to next degree if last ind is 1

    elif last_ind == 1:
        inc_ind = nind
        t = next_monomial[0] + 1

    # Else increment

    else:
        inc_ind = last_ind - 1
        t = next_monomial[inc_ind]

    next_monomial[last_ind - 1] = 0
    next_monomial[inc_ind - 1] = next_monomial[inc_ind - 1] + 1
    next_monomial[nind - 1] = next_monomial[nind - 1] + t - 1

    return next_monomial


def grlex_ordering_to_deg(nind: int, max_deg: int) -> list:
    """
    Generates all monomials in ascending graded lexicographical ordering up
    to a provided maximum degree.

    :param nind: The number of independent variables
    :type nind: int
    :param max_deg: The maximum degree of the monomials
    :type max_deg: int

    :return: A list of all monomials up to specified max degree
    :rtype: list
    """

    # Start list

    monomials = []
    monomial = [0] * nind
    monomials.append(monomial)

    # Generate next until at max degree

    current_degree = 0
    while current_degree <= max_deg:

        monomial = next_grlex(monomial, nind)
        current_degree = sum(monomial)

        if current_degree <= max_deg:
            monomials.append(monomial)

    # Return list

    return monomials


def next_grrevlex(monomial: list, nind: int) -> list:
    """
    Generates the next monomial in the graded reverse lexicographical ordering.

    :param monomial: The current monomial represented as a list of nonnegative integers
    :type monomial: list
    :param nind: The number of independent variables
    :type nind: int

    :return: The next monomial in the ordering represented as a list of nonnegative integers
    :rtype: list
    """
    next_monomial = monomial.copy()

    # Special case: single variable
    if nind == 1:
        next_monomial[0] += 1
        return next_monomial

    # Special case: degree 0 -> move to first degree-1 monomial
    if all(x == 0 for x in monomial):
        next_monomial[0] = 1
        return next_monomial

    # Find the rightmost nonzero position that is NOT the last position
    # (positions 0 through nind-2)
    rightmost_movable = -1
    for i in range(nind - 2, 0, -1):
        if next_monomial[i] > 0:
            rightmost_movable = i
            break

    if rightmost_movable > 0:
        # Move one unit from rightmost_movable to rightmost_movable + 1
        # and collect everything from positions > rightmost_movable + 1 back to position 1
        carry = sum(next_monomial[rightmost_movable + 1:])
        for i in range(rightmost_movable + 1, nind):
            next_monomial[i] = 0
        next_monomial[rightmost_movable] -= 1
        next_monomial[rightmost_movable + 1] = carry + 1
    else:
        # No movable position found in indices 1..nind-2
        # All nonzero values are at position 0 and/or the last position
        t = next_monomial[0]
        z = next_monomial[nind - 1]

        if t == 0 and z == 0:
            # Shouldn't reach here if monomial isn't all zeros (handled above)
            next_monomial[0] = 1
        elif z > 0:
            # Last position has value: rollover
            next_monomial[nind - 1] = 0
            if t == 0:
                # Only last position had value: increment total degree
                next_monomial[0] = z + 1
            else:
                # Both position 0 and last position: redistribute
                next_monomial[0] = t - 1
                next_monomial[1] = z + 1
        else:
            # Only position 0 has value (z == 0, t > 0): start distributing
            next_monomial[0] = t - 1
            next_monomial[1] = 1

    return next_monomial


def grrevlex_ordering_to_deg(nind: int, max_deg: int) -> list:
    """
    Generates all monomials in ascending graded reverse lexicographical ordering up
    to a provided maximum degree.

    :param nind: The number of independent variables
    :type nind: int
    :param max_deg: The maximum degree of the monomials
    :type max_deg: int

    :return: A list of all monomials up to specified max degree
    :rtype: list
    """

    # Start list

    monomials = []
    monomial = [0] * nind
    monomials.append(monomial)

    # Generate next until at max degree

    current_degree = 0
    while current_degree <= max_deg:

        monomial = next_grrevlex(monomial, nind)
        current_degree = sum(monomial)

        if current_degree <= max_deg:
            monomials.append(monomial)

    # Return list

    return monomials
