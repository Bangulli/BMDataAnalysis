import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r, vectors
pandas2ri.activate()
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from sklearn.base import BaseEstimator, ClusterMixin
from .utils import wide_to_long
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import globalenv
from rpy2.robjects.vectors import IntVector



class HLMEClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.posteriors_ = None
        self.base = importr('base')
        self.lcmm = importr('lcmm')
        self.gs = importr('lcmm').gridsearch

    def fit(self, X, y=None):
        X = wide_to_long(X)
        X['time'] /= 360
        print(X)
        # Expect X to be a pandas DataFrame with columns: subject, time, value
        with localconverter(robjects.default_converter + pandas2ri.converter):
            rdf = pandas2ri.py2rpy(X)

        r.assign("rdf", rdf)

        init = self.lcmm.hlme(
            fixed = Formula('value ~ time'),
            random = Formula('~ 1'),
            subject = 'subject',
            ng = 1,
            data = rdf,
        )
        print(f"Initialization model converged: {init.rx2("conv")}")

        # Step 2: Create the unevaluated R expression for the model
        r_expr = robjects.r(f'''
            quote(
                hlme(fixed = value ~ time,
                    mixture = ~1,
                    random = ~1,
                    subject = "subject",
                    ng = {self.n_components},
                    data = rdf,
                    nwg = FALSE)
            )
        ''')

        # Step 3: Run gridsearch with unevaluated R expression
        self.model = self.gs(
            m=r_expr,           # pass the quoted model expression
            rep=100,
            maxiter=500,
            minit=init
        )

        print(f"Initialization model converged: {self.model.rx2("conv")}")
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


class LCMMClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.posteriors_ = None
        self.base = importr("base")
        self.lcmm = importr('lcmm')
        self.gs = importr('lcmm').gridsearch

    def fit(self, X, y=None):
        X = wide_to_long(X)
        X["time"] /= 360

        with localconverter(robjects.default_converter + pandas2ri.converter):
            rdf = pandas2ri.py2rpy(X)

        r.assign("rdf", rdf)

        # Step 1: Initial model with ng=1
        init = self.lcmm.lcmm(
            fixed=robjects.Formula("value ~ 1 + time"),
            random=robjects.Formula("~ 1"),
            subject="subject",
            ng=1,
            data=rdf
        )
        print(f"Initialization model converged: {init.rx2('conv')}")

        base = importr("base")

        model_call = base.call(
            "lcmm",
            fixed=Formula("value ~ 1 + time"),
            mixture=Formula("~ 1"),
            random=Formula("~ 1"),
            subject="subject",
            ng=self.n_components,
            data=r["rdf"],
            nwg=False,
            link="5-quant-splines"
        )
        # Step 3: Run gridsearch
        self.model = self.gs(
            m=model_call,
            rep=100,
            maxiter=500,
            minit=init
        )

        print(f"Final model convergence status: {self.model.rx2('conv')}")
        self.posteriors_ = self.model.rx2('pprob')
        return self


    def predict(self, X):
        # Return most likely class for each subject
        with localconverter(robjects.default_converter + pandas2ri.converter):
            post = pandas2ri.rpy2py(self.posteriors_)
        return post['class'].values
    
    def score(self, X, y):
        return 0
