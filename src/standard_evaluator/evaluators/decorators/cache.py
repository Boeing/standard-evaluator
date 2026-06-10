import sqlite3
from typing import Callable

import pandas as pd

from standard_evaluator.utilities import concat_w_empty
from standard_evaluator.problem import OptProblem


class cacher:
    """Add caching ability to an evaluator. When evaluating sites the cache
    will be checked to determine if any of the sites have already been evaluated.
    If so, then the cached sites will take their response values from the cache.
    The remaining, uncached sites will be evaluated and added to the cache. The
    cache operates using an underlying SQLite database.
    """

    def __init__(
        self,
        func: Callable,
        opt_problem: OptProblem,
        db_name: str,
        table_name: str = "Design_Data",
        delta: float = 1e-8,
        auto_update: bool = False,
    ) -> None:
        """Initialize the cacher.

        Args:
            func: Function to apply caching to.
            opt_problem: Problem definition as an OptProblem instance.
            db_name: Name of database file to read/write cached sites.
            table_name: Name of the table where site data is stored.
                Defaults to 'Design_Data'.
            delta: Unscaled distance used to determine if two sites are
                duplicate. Defaults to 1e-8.
            auto_update: When True the database will be re-read before
                evaluating sites. Defaults to False.
        """
        self._func = func
        self._opt_problem = opt_problem

        # Extract variable and response names as strings
        self._variables = [var.name for var in self._opt_problem.variables]
        self._responses = [resp.name for resp in self._opt_problem.responses]

        # Store the number of independents and dependents
        self._num_independent = len(self._variables)
        self._num_dependent = len(self._responses)

        # Build dtypes dict with proper pandas dtypes
        self._dtypes = {}
        var_resp_list = self._opt_problem.variables + self._opt_problem.responses
        for var in var_resp_list:
            if var.class_type == "cat":
                self._dtypes[var.name] = pd.CategoricalDtype(
                    categories=var.bounds, ordered=True
                )
            elif var.class_type == "int":
                self._dtypes[var.name] = "Int64"
            else:
                self._dtypes[var.name] = "float64"

        # Store parameters
        self._delta_sq = delta**2
        self._auto_update = auto_update

         # Store database info
        self._table_name = table_name

        # Create empty cache DataFrame with correct dtypes
        self._cache = pd.DataFrame(
            {name: pd.Series(dtype=dtype) for name, dtype in self._dtypes.items()}
        )

        # Connect to database
        self._connection = sqlite3.connect(db_name)
        _cursor = self._connection.cursor()

        # Get the number of tables in the database
        n_tables = _cursor.execute(
            "SELECT count(name) "
            "FROM sqlite_master "
            f"WHERE type = 'table' AND name = '{table_name}';"
        ).fetchall()

        if n_tables == [(0,)]:
            # If there are no tables create an empty one
            self._cache.to_sql(table_name, self._connection, index=False)
        else:
            # Read the database otherwise
            self._read_db()

    def __call__(self, sites: pd.DataFrame) -> None:
        """Evaluate sites using the cache for previously evaluated sites.

        Args:
            sites: Sites to evaluate.
        """
        # Read database if auto update is enabled
        if self._auto_update:
            self._read_db()

        # Convert categorical columns in sites if they exist
        for col, dtype in self._dtypes.items():
            if col in sites.columns and isinstance(dtype, pd.CategoricalDtype):
                sites[col] = pd.Categorical(
                    sites[col], categories=dtype.categories, ordered=dtype.ordered
                )
        # Whether or not a site is cached
        point_cached = [False] * len(sites)
        # Indices of sites that are in the cache
        closest_points = []
        # Determine which sites are cached
        for i in range(len(sites)):
            if self._cache.empty:
                break
            # Compute squared distance between site and cached sites
            distances = (
                ((self._cache[self._variables] - sites.iloc[i][self._variables]) ** 2)
                .sum(axis=1)
                .astype(float)
            )
            # Determine if site is too close to any cached sites
            if distances.min() < self._delta_sq:
                # If so, mark it as cached and store its index
                point_cached[i] = True
                closest_points.append(distances.idxmin())

        if any(point_cached):
            # Convert cached categorical columns before assignment
            for col, dtype in self._dtypes.items():
                if col in self._cache.columns and isinstance(dtype, pd.CategoricalDtype):
                    self._cache[col] = pd.Categorical(
                        self._cache[col], categories=dtype.categories, ordered=dtype.ordered
                    )
            # Fill in cached response values                    
            sites.loc[point_cached, self._responses] = self._cache.loc[
                closest_points, self._responses
            ].to_numpy()

        # Invert point_cached
        not_cached = [not cached for cached in point_cached]
        
        # Evaluate and cache uncached sites
        if any(not_cached):
            # Get sites that need to be evaluated
            need_to_eval = sites.loc[not_cached, self._variables]

            # Evaluate sites
            self._func(need_to_eval)

            # Save sites to database
            need_to_eval[self._variables + self._responses].to_sql(
                self._table_name, self._connection, if_exists="append", index=False
            )
            self._connection.commit()

            # Update local cache
            self._cache = concat_w_empty([self._cache, need_to_eval])

            # Update input dataframe with response values
            sites.loc[not_cached, self._responses] = need_to_eval.loc[:, self._responses]

    def get_cache(self) -> pd.DataFrame:
        """Return a copy of the current cache."""
        return self._cache.copy(deep=True)

    def _read_db(self):
        """Read cached sites from the database and apply correct dtypes."""
        # Read database and convert column types. This is really only needed
        # with mixed column types. Especially categorical columns
        df = pd.read_sql(f"SELECT * FROM {self._table_name}", self._connection)
        for col, dtype in self._dtypes.items():
            if isinstance(dtype, pd.CategoricalDtype):
                df[col] = pd.Categorical(df[col], categories=dtype.categories, ordered=dtype.ordered)
            else:
                df[col] = df[col].astype(dtype)
        self._cache = df

    def __del__(self):
        # Close connection to database
        self._connection.close()
