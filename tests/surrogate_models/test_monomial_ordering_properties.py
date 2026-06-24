"""Property-based test for monomial ordering correctness.

Feature: surrogate-model-migration
Property 1: Monomial ordering correctness

**Validates: Requirements 3.3, 3.4**
"""

from math import comb

from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.surrogate_models.polynomial_model_utils.monomial_ordering import (
    grlex_ordering_to_deg,
    grrevlex_ordering_to_deg,
)


# Strategies for nind (1-5) and max_deg (0-4)
nind_strategy = st.integers(min_value=1, max_value=5)
max_deg_strategy = st.integers(min_value=0, max_value=4)


@given(nind=nind_strategy, max_deg=max_deg_strategy)
@settings(max_examples=100)
def test_grlex_ordering_correctness(nind: int, max_deg: int):
    """Property 1: For any valid nind >= 1 and max_deg >= 0,
    grlex_ordering_to_deg produces a list that:
    - Begins with the zero-degree monomial [0, 0, ..., 0]
    - Contains exactly C(nind + max_deg, max_deg) monomials
    - Has non-decreasing total degree across the list
    - Contains no duplicate entries

    Feature: surrogate-model-migration
    Property 1: Monomial ordering correctness

    **Validates: Requirements 3.3, 3.4**
    """
    monomials = grlex_ordering_to_deg(nind, max_deg)

    # Begins with the zero monomial
    assert monomials[0] == [0] * nind, (
        f"First monomial should be zero monomial, got {monomials[0]}"
    )

    # Correct count: C(nind + max_deg, max_deg)
    expected_count = comb(nind + max_deg, max_deg)
    assert len(monomials) == expected_count, (
        f"Expected {expected_count} monomials for nind={nind}, max_deg={max_deg}, "
        f"got {len(monomials)}"
    )

    # Non-decreasing total degree
    degrees = [sum(m) for m in monomials]
    for i in range(1, len(degrees)):
        assert degrees[i] >= degrees[i - 1], (
            f"Total degree decreased at index {i}: "
            f"degree {degrees[i-1]} -> {degrees[i]}, "
            f"monomials: {monomials[i-1]} -> {monomials[i]}"
        )

    # No duplicates
    tuples = [tuple(m) for m in monomials]
    assert len(set(tuples)) == len(tuples), (
        f"Duplicate monomials found in grlex ordering for nind={nind}, max_deg={max_deg}"
    )


@given(nind=nind_strategy, max_deg=max_deg_strategy)
@settings(max_examples=100)
def test_grrevlex_ordering_correctness(nind: int, max_deg: int):
    """Property 1: For any valid nind >= 1 and max_deg >= 0,
    grrevlex_ordering_to_deg produces a list that:
    - Begins with the zero-degree monomial [0, 0, ..., 0]
    - Contains exactly C(nind + max_deg, max_deg) monomials
    - Has non-decreasing total degree across the list
    - Contains no duplicate entries

    Feature: surrogate-model-migration
    Property 1: Monomial ordering correctness

    **Validates: Requirements 3.3, 3.4**
    """
    monomials = grrevlex_ordering_to_deg(nind, max_deg)

    # Begins with the zero monomial
    assert monomials[0] == [0] * nind, (
        f"First monomial should be zero monomial, got {monomials[0]}"
    )

    # Correct count: C(nind + max_deg, max_deg)
    expected_count = comb(nind + max_deg, max_deg)
    assert len(monomials) == expected_count, (
        f"Expected {expected_count} monomials for nind={nind}, max_deg={max_deg}, "
        f"got {len(monomials)}"
    )

    # Non-decreasing total degree
    degrees = [sum(m) for m in monomials]
    for i in range(1, len(degrees)):
        assert degrees[i] >= degrees[i - 1], (
            f"Total degree decreased at index {i}: "
            f"degree {degrees[i-1]} -> {degrees[i]}, "
            f"monomials: {monomials[i-1]} -> {monomials[i]}"
        )

    # No duplicates
    tuples = [tuple(m) for m in monomials]
    assert len(set(tuples)) == len(tuples), (
        f"Duplicate monomials found in grrevlex ordering for nind={nind}, max_deg={max_deg}"
    )
