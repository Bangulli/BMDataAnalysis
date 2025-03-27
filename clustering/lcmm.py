import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r, vectors
pandas2ri.activate()
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from sklearn.base import BaseEstimator, ClusterMixin
from .utils import wide_to_long


class HLMEClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.posteriors_ = None
        self.lcmm = rpackages.importr('lcmm')

    def fit(self, X, y=None):
        # Expect X to be a pandas DataFrame with columns: subject, time, value
        with localconverter(robjects.default_converter + pandas2ri.converter):
            rdf = pandas2ri.py2rpy(wide_to_long(X))

        r.assign("rdf", rdf)

        init = self.lcmm.hlme(
            fixed = robjects.Formula('value ~ time'),
            random = robjects.Formula('~ time'),
            subject = 'subject',
            ng = 1,
            data = rdf
        )

        self.model = self.lcmm.hlme(
            fixed = robjects.Formula('value ~ time'),
            mixture = robjects.Formula('~ time'),
            random = robjects.Formula('~ time'),
            subject = 'subject',
            ng = self.n_components,
            data = rdf,
            B = init
        )

        # Store posteriors
        self.posteriors_ = self.model.rx2('pprob')


        return self

    def predict(self, X):
        # Return most likely class for each subject
        with localconverter(robjects.default_converter + pandas2ri.converter):
            post = pandas2ri.rpy2py(self.posteriors_)
        return post['class'].values
    
    def score(self, X, y):
        return 0


def run_hlme(df):
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # Choose first CRAN mirror

    # Install lcmm (or any other R package)
    package_name = 'lcmm'
    if not rpackages.isinstalled(package_name):
        utils.install_packages(package_name)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(wide_to_long(df))

    r.assign("rdf", rdf)  

    r('''
        model <- hlme(fixed = value ~ time,
                    random = ~ time,
                    subject = 'subject',
                    ng = 2,
                    data = rdf)

        summary(model)
        ''')
    
    posteriors = r('model$pprob')

    with localconverter(robjects.default_converter + pandas2ri.converter):
        posteriors_df = robjects.conversion.rpy2py(posteriors)

    return posteriors_df

def run_lcmm(df):
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # Choose first CRAN mirror

    # Install lcmm (or any other R package)
    package_name = 'lcmm'
    if not rpackages.isinstalled(package_name):
        utils.install_packages(package_name)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(wide_to_long(df))

    r.assign("rdf", rdf)  

    r('''
        model <- lcmm(fixed = value ~ time,
                    random = ~ time,
                    subject = 'subject',
                    ng = 2,
                    data = rdf)

        summary(model)
        ''')
    
    posteriors = r('model$pprob')

    with localconverter(robjects.default_converter + pandas2ri.converter):
        posteriors_df = robjects.conversion.rpy2py(posteriors)

    return posteriors_df