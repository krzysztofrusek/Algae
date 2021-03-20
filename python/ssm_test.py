import unittest
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

tfd=tfp.distributions
tfb=tfp.bijectors
sts = tfp.sts
from numpy.polynomial import polynomial as P
import pandas as pd

from ssm import ssm

class TransitionTest(unittest.TestCase):
    def test_matmul(self):
        F=np.random.normal(size=(18,18))

        top = tf.concat([F @ F, F], axis=1)
        bottom = tf.zeros_like(top)

        F2 = tf.concat([top, bottom], axis=0)

        dense = tf.linalg.LinearOperatorFullMatrix(F2)
        operator = ssm.TransitionOperator(tf.linalg.LinearOperatorFullMatrix(F))

        for s in [
            (1,36),
            (2,1,36),
            (36,36),
            (2,36,36),
            (36,1),
            (2,36,1)
        ]:
            for a in [True, False]:
                for b in [True, False]:
                    x = np.random.normal(size=s)
                    try:
                        y1=dense.matmul(x,a,b)
                    except:
                        # wrong shape
                        continue
                    y2=operator.matmul(x,a,b)
                    self.assertTrue(np.allclose(y1,y2))



class ObservationTest(unittest.TestCase):
    def test_matmul(self):
        F = np.random.normal(size=(18, 18))
        H = np.random.normal(size=(1, 18))
        H2 = tf.concat([H + H @ F, H], axis=1)
        H2= H2*tf.constant(0.5, dtype=H2.dtype)

        dense = tf.linalg.LinearOperatorFullMatrix(H2)
        operator = ssm.ObservationOperator(tf.linalg.LinearOperatorFullMatrix(H),tf.linalg.LinearOperatorFullMatrix(F))

        for s in [
            (1,36),
            (2,1,36),
            (36,36),
            (2,36,36),
            (36,1),
            (2,36,1)
        ]:
            for a in [True, False]:
                for b in [True, False]:
                    x = np.random.normal(size=s)
                    try:
                        y1=dense.matmul(x,a,b)
                    except:
                        # wrong shape
                        continue
                    y2=operator.matmul(x,a,b)
                    self.assertTrue(np.allclose(y1,y2))


class SSMCyfronetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        #super(SSMCyfronetTestCase, self).__init__(*args, **kwargs)
        df = pd.read_csv("data/agh_core2h.csv", index_col='timestamp')
        df.index = pd.to_datetime(df.index)
        df.sort_values(by=['timestamp'], ascending=True)
        self.cyfronet = df['ucirtr-cyfronet'].to_numpy()
    @property
    def model(self):
        observed_time_series=self.cyfronet
        seasonal_weakly = tfp.sts.Seasonal(
            num_seasons=7,
            num_steps_per_season=12,
            observed_time_series=observed_time_series, name="weakly")
        seasonal_dayly = tfp.sts.Seasonal(
            num_seasons=12, observed_time_series=observed_time_series, name="dayly")
        autoregressive = sts.Autoregressive(
            order=1,
            observed_time_series=observed_time_series,
            name='autoregressive')

        model = sts.Sum([autoregressive, seasonal_weakly, seasonal_dayly], observed_time_series=observed_time_series)
        #model = sts.Sum([autoregressive], observed_time_series=observed_time_series)
        return model

    @property
    def dist(self):
        param_vals = []

        for param in self.model.parameters:
            param_val = param.prior.sample()
            param_vals.append(param_val)

        cyfronet_dist = self.model.make_state_space_model(10, param_vals)
        return cyfronet_dist

    def test_matrix(self):

        dist = self.dist
        agg_cyfronet = ssm.aggregated_model_dense(dist)
        samples = agg_cyfronet.sample(2)
        lp=agg_cyfronet.log_prob(samples)
        agg_cyfronet = ssm.aggregated_model_operator(dist)
        lpo=agg_cyfronet.log_prob(samples)
        self.assertTrue(np.allclose(lp, lpo, atol=0.1))
        agg_cyfronet.sample(2)
        return

    def test_agg(self):
        file: str = "data/agh_core30min.csv"
        df = pd.read_csv(file, index_col='timestamp')
        df.index = pd.to_datetime(df.index)
        df.sort_values(by=['timestamp'], ascending=True)
        ts = df['cyfronet-ucirtr']

        file:str="data/agh_core2h.csv"
        df = pd.read_csv(file, index_col='timestamp')
        df.index = pd.to_datetime(df.index)
        df.sort_values(by=['timestamp'], ascending=True)
        ts2 = df['cyfronet-ucirtr']

        p = ssm.AggregatedPredictor(num_forecast_steps=4 * 2 * 24, ts=ts.astype(np.float32), ts2=ts2.astype(np.float32))
        print(p.fit(2))
        print(p.q_samples)
        pass


if __name__ == '__main__':
    unittest.main()
