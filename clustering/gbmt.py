import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages

from rpy2.robjects.methods import RS4
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri, Formula, r
from rpy2.robjects.packages import importr

# Activate pandas-R conversion
pandas2ri.activate()

# Import flexmix from R
flexmix = importr('flexmix')


class FlexmixClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, formula="value ~ time"):
        self.n_clusters = n_clusters
        self.formula = formula
        self.model_ = None
        self.assignments_ = None

    def _wide_to_long(self, df):
        df_long = df.copy()
        df_long["subject"] = df_long.index
        df_long = df_long.melt(id_vars="subject", var_name="time", value_name="value")
        df_long["time"] = df_long["time"].astype(float)
        return df_long

    def fit(self, X, y=None):
        
        df_long = self._wide_to_long(X)
        df_long['time'] /= 360

        with localconverter(robjects.default_converter + pandas2ri.converter):
            rdf = pandas2ri.py2rpy(df_long)

        robjects.r.assign("rdf", rdf)
        robjects.r.assign("k", self.n_clusters)

        r_code = f"""
        library(flexmix)
        rdf$subject <- as.factor(rdf$subject)
        model <- flexmix({self.formula}, data=rdf, k=k, nrep=50)
        """
        robjects.r(r_code)

        self.model_ = robjects.r["model"]
        print(self.model_.do_slot("components"))
        if isinstance(self.model_, RS4):
            self.assignments_ = self.model_.do_slot("cluster")
        else:
            raise ValueError("Expected an RS4 object for flexmix model")

        return self

    def predict(self, X):
        # Convert R cluster vector to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            full_assignments = pandas2ri.rpy2py(self.assignments_)  # length = n_timepoints_total

        # Create long-format index to group by subject
        df_long = self._wide_to_long(X)
        df_long["cluster"] = full_assignments

        # Return one cluster per subject (e.g., the most frequent)
        subject_clusters = df_long.groupby("subject")["cluster"].agg(lambda x: x.mode()[0])
        return subject_clusters.reindex(X.index).values  # align to original wide index



def cluster_with_flexmix(df, n_clusters=3):
    """
    Clusters time-series data using flexmix via rpy2.

    Parameters:
        df (pd.DataFrame): DataFrame (739, 6) with columns ['60', '120', '180', '240', '300', '360']
        n_clusters (int): Number of desired clusters.

    Returns:
        np.ndarray: Cluster assignments for each of the 739 samples.
    """

    # Prepare DataFrame to long format required by flexmix
    df_long = df.copy()
    df_long['id'] = df_long.index.astype(str)
    df_long = df_long.melt(id_vars=['id'], var_name='day', value_name='value')
    df_long['day'] = df_long['day'].astype(int)

    sclr = StandardScaler()
    df_long = pd.DataFrame(sclr.fit_transform(df_long), columns=df_long.columns)

    # Convert to R DataFrame
    rdf = pandas2ri.py2rpy(df_long)

    # Define flexmix formula (using day as predictor)
    formula = Formula('value ~ day | id')

    # Fit the flexmix model (no initial clusters)
    model = flexmix.flexmix(formula, data=rdf, k=n_clusters)

    # Retrieve cluster assignments for each id
    clusters = r['clusters'](model, newdata=rdf)

    # Extract unique cluster assignments per sample id
    cluster_assignments = np.array(pd.Series(clusters).groupby(df_long['id']).first())

    return cluster_assignments

# Example usage:
# clusters = cluster_with_flexmix(df, n_clusters=4)
# print(clusters)
