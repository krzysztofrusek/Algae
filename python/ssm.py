from dataclasses import dataclass
from typing import Any
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
sts = tfp.sts


def aggregated_model_simple(fine_model):
    F = fine_model.transition_matrix(0).to_dense()

    H = fine_model.observation_matrix(0).to_dense()

    transition_noise = tfd.MultivariateNormalLinearOperator(
        scale=tf.linalg.LinearOperatorBlockDiag(2 * [fine_model.transition_noise(0).scale]))

    num_rows = fine_model.observation_noise(0).scale.shape[0]
    # observation_noise=tfd.MultivariateNormalLinearOperator(scale=ar2.observation_noise.scale.matmul(tf.linalg.LinearOperatorScaledIdentity(num_rows, tf.cast(0.5,ar2.dtype))))
    observation_noise = tfd.MultivariateNormalLinearOperator(scale=fine_model.observation_noise(0).scale.matmul(
        tf.linalg.LinearOperatorScaledIdentity(num_rows, tf.cast(np.sqrt(0.5), fine_model.dtype))))

    # z_shape = ar_model.initial_state_prior.event_shape
    initial_state_prior = tfd.MultivariateNormalLinearOperator(
        scale=tf.linalg.LinearOperatorBlockDiag(2 * [fine_model.initial_state_prior.scale]))

    # transition_matrix = tf.linalg.LinearOperatorBlockDiag([
    #                                                        tf.linalg.LinearOperatorFullMatrix(F@F),
    #                                                        tf.linalg.LinearOperatorFullMatrix(F+F@F)])
    top = tf.concat([F @ F, F], axis=1)
    bottom = tf.zeros_like(top)

    transition_matrix = tf.concat([top, bottom], axis=0)

    observation_matrix = tf.concat([H + H @ F, H], axis=1) * tf.constant(0.5, dtype=fine_model.dtype)

    agg_ar = tfd.LinearGaussianStateSpaceModel(
        # num_timesteps=int(ntime / 2),
        num_timesteps=4,
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        initial_state_prior=initial_state_prior
    )
    return agg_ar


class TransitionOperator(tf.linalg.LinearOperator):
    def __init__(self,
                 operator,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 name=None):
        self.transition_matrix_ = operator
        super(TransitionOperator, self).__init__(
            dtype=operator.dtype,
            graph_parents=None,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=True,
            name=name)

    def _shape(self):
        return tf.TensorShape([2 * s for s in self.transition_matrix_.shape])

    #@tf.function(experimental_compile=True)
    def _matmul(self, x, adjoint=False, adjoint_arg=False):

        F = self.transition_matrix_
        FF = F @ F

        s = x.shape
        batched = len(s) > 2

        transpose_list = [0, 2, 1] if batched else [1, 0]
        x = tf.transpose(x, transpose_list) if adjoint_arg else x
        # split matrix along rows
        t, b = tf.split(x, 2, axis=-2)
        concat_axis = 1 if batched else 0
        if adjoint:
            parts = [FF.matmul(t, adjoint=True), F.matmul(t, adjoint=True)]
        else:
            ret = FF.matmul(t) + F.matmul(b)
            zero = tf.zeros_like(ret)
            parts = [ret, zero]

        return tf.concat(parts, axis=concat_axis)


class ObservationOperator(tf.linalg.LinearOperator):
    def __init__(self,
                 operator_H,
                 operator_F,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 name=None):
        self.observation_matrix_ = operator_H
        self.transition_matrix_ = operator_F
        super(ObservationOperator, self).__init__(
            dtype=operator_H.dtype,
            graph_parents=None,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=False,
            name=name)

    def _shape(self):
        return tf.TensorShape([self.observation_matrix_.shape[0],
                               2 * self.observation_matrix_.shape[1]])

    #@tf.function(experimental_compile=True)
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        H = self.observation_matrix_
        F = self.transition_matrix_
        HF = H @ F

        s = x.shape
        batched = len(s) > 2

        transpose_list = [0, 2, 1] if batched else [1, 0]
        x = tf.transpose(x, transpose_list) if adjoint_arg else x
        scale = tf.constant(0.5, dtype=x.dtype)
        if adjoint:
            l, r = tf.split(x, 2, axis=-1)
            t = [H.matmul(l, adjoint=True) + HF.matmul(l, adjoint=True),
                 H.matmul(r, adjoint=True) + HF.matmul(r, adjoint=True)]
            b = [H.matmul(l, adjoint=True), H.matmul(r, adjoint=True)]
            concat_axis = 2 if batched else 1
            t = tf.concat(t, axis=concat_axis)
            b = tf.concat(b, axis=concat_axis)
            concat_axis = 1 if batched else 0
            return scale * tf.concat([t, b], concat_axis)
        else:
            t, b = tf.split(x, 2, axis=-2)
            return scale * (H @ t + HF @ t + H @ b)

def functionalize(arg):
    ret = arg if callable(arg) else lambda t: arg
    return ret

def aggregated_model_operator(fine_model, num_timesteps=4):
    F = functionalize(fine_model.transition_matrix)
    H = functionalize(fine_model.observation_matrix)
    observation_noise = functionalize(fine_model.observation_noise)
    transition_noise = functionalize(fine_model.transition_noise)


    def transition_noise_fn(t):
        return tfd.MultivariateNormalLinearOperator(
            scale=tf.linalg.LinearOperatorBlockDiag(2 * [transition_noise(t).scale]))

    num_rows = fine_model.event_shape[1]
    def observation_noise_fn(t):
        return tfd.MultivariateNormalLinearOperator(
            scale=observation_noise(t).scale.matmul(
                tf.linalg.LinearOperatorScaledIdentity(num_rows, tf.cast(np.sqrt(0.5), fine_model.dtype))))

    initial_state_prior = tfd.MultivariateNormalLinearOperator(
        scale=tf.linalg.LinearOperatorBlockDiag(2 * [fine_model.initial_state_prior.scale]))

    def transition_matrix_fn(t):
        return TransitionOperator(F(t))

    def observation_matrix_fn(t):
        return ObservationOperator(H(t), F(t))

    agg_ar = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=transition_matrix_fn,
        transition_noise=transition_noise_fn,
        observation_matrix=observation_matrix_fn,
        observation_noise=observation_noise_fn,
        initial_state_prior=initial_state_prior
    )
    return agg_ar
def aggregated_model_dense(fine_model, num_timesteps=4):
    F = functionalize(fine_model.transition_matrix)
    H = functionalize(fine_model.observation_matrix)
    observation_noise = functionalize(fine_model.observation_noise)
    transition_noise = functionalize(fine_model.transition_noise)


    def transition_noise_fn(t):
        return tfd.MultivariateNormalLinearOperator(
            scale=tf.linalg.LinearOperatorBlockDiag(2 * [transition_noise(t).scale]))

    num_rows = fine_model.event_shape[1]
    def observation_noise_fn(t):
        return tfd.MultivariateNormalLinearOperator(
            scale=observation_noise(t).scale.matmul(
                tf.linalg.LinearOperatorScaledIdentity(num_rows, tf.cast(np.sqrt(0.5), fine_model.dtype))))

    initial_state_prior = tfd.MultivariateNormalLinearOperator(
        scale=tf.linalg.LinearOperatorBlockDiag(2 * [fine_model.initial_state_prior.scale]))

    def transition_matrix_fn(t):
        _F=F(t).to_dense()
        top = tf.concat([_F @ _F, _F], axis=1)
        bottom = tf.zeros_like(top)

        return tf.linalg.LinearOperatorFullMatrix(tf.concat([top, bottom], axis=0))

    def observation_matrix_fn(t):
        _F=F(t).to_dense()
        _H = H(t).to_dense()
        return tf.linalg.LinearOperatorFullMatrix(tf.concat([_H + _H @ _F, _H], axis=1) * tf.constant(0.5, dtype=fine_model.dtype))

    agg_ar = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=transition_matrix_fn,
        transition_noise=transition_noise_fn,
        observation_matrix=observation_matrix_fn,
        observation_noise=observation_noise_fn,
        initial_state_prior=initial_state_prior
    )
    return agg_ar


@dataclass
class TimeSeriesPredictor:
    num_forecast_steps: int
    ts: pd.Series
    posterior: Any = None

    @property
    def train_ts(self):
        return self.ts[:-self.num_forecast_steps].to_numpy()

    def model(self):

        trend = sts.LocalLinearTrend(observed_time_series=self.train_ts)
        freq = list(range(1,17))
        seasonal_dayly = tfp.sts.SmoothSeasonal(period=48,
                                                frequency_multipliers=freq,
                                                observed_time_series=self.train_ts,
                                                name="dayly")
        seasonal_weakly = tfp.sts.SmoothSeasonal(period=48*7,
                                                frequency_multipliers=freq,
                                                observed_time_series=self.train_ts,
                                                name="weakly")


        autoregressive = sts.Autoregressive(
            order=1,
            observed_time_series=self.train_ts,
            name='autoregressive')

        model = sts.Sum([trend, seasonal_weakly, seasonal_dayly, autoregressive], observed_time_series=self.train_ts)
        return model

    def fit(self, num_variational_steps: int = 200):
        model = self.model()
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
            model=model)
        num_variational_steps = int(num_variational_steps)

        optimizer = tf.optimizers.Adam(learning_rate=.1)



        # Using fit_surrogate_posterior to build and optimize the variational loss function.
        @tf.function(autograph=False, experimental_compile=True)
        def train():
            elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=model.joint_log_prob(
                    observed_time_series=self.train_ts),
                surrogate_posterior=variational_posteriors,
                optimizer=optimizer,
                num_steps=num_variational_steps)
            return elbo_loss_curve

        elbo_loss_curve = train()
        self.posterior = variational_posteriors
        return elbo_loss_curve

        #self.q_samples = variational_posteriors.sample(num_posterior)

        #plt.plot(elbo_loss_curve)
        #self._save(variational_posteriors)

        #plt.show()
    def save(self, directory=os.path.dirname(__file__)):
        m=tf.Module()
        m.posterior=self.posterior.variables
        checkpoint = tf.train.Checkpoint(posterior=m)
        checkpoint.write(os.path.join(directory,self.__class__.__name__))

    def load(self, directory=os.path.dirname(__file__)):
        self.posterior = tfp.sts.build_factored_surrogate_posterior( model=self.model())
        m=tf.Module()
        m.posterior=self.posterior.variables
        checkpoint = tf.train.Checkpoint(posterior=m)
        checkpoint.read(os.path.join(directory,self.__class__.__name__))


    def predict(self, num_posterior: int = 50):
        forecast_dist = tfp.sts.forecast(
            self.model(),
            observed_time_series=self.train_ts,
            parameter_samples=self.posterior.sample(num_posterior),
            num_steps_forecast=self.num_forecast_steps)


        return(
            forecast_dist.mean().numpy()[..., 0],
            forecast_dist.stddev().numpy()[..., 0])





Root = tfd.JointDistributionCoroutine.Root

@dataclass
class AggregatedPredictor(TimeSeriesPredictor):
    ts2: pd.Series=None

    @property
    def train_ts2(self):
        return self.ts2.to_numpy()


    def fit(self, num_variational_steps: int = 200, num_posterior: int = 50):
        '''
        :param num_variational_steps:
        :param num_posterior:
        :return:
        '''

        model = self.model()

        nts1= self.train_ts.shape[0]
        nts2 = self.train_ts2.shape[0]
        train_ts = tf.convert_to_tensor(self.train_ts[..., np.newaxis])
        train_ts2 = tf.convert_to_tensor(self.train_ts2[..., np.newaxis])
        def sts_model():
            # Encode the parameters of the STS model as random variables.
            param_vals = []
            for param in model.parameters:
                param_val = yield Root(param.prior)
                param_vals.append(param_val)


            dist = model.make_state_space_model(nts1, param_vals)
            dist_up2 = aggregated_model_operator(dist, nts2)
            yield dist
            yield aggregated_model_operator(dist_up2, nts2)

        jd= tfd.JointDistributionCoroutine(sts_model)

        constraining_bijectors = ([param.bijector for param in model.parameters] +
                                  2*[tfb.Identity()])

        target_log_prob_fn = lambda *args: jd.log_prob(args + (train_ts,train_ts2))
        variational_posteriors = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=jd.event_shape[:-2],
        constraining_bijectors=constraining_bijectors)


        optimizer = tf.optimizers.Adam(learning_rate=.1)


        @tf.function(autograph=False, experimental_compile=True)
        def train():
            elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=target_log_prob_fn,
                surrogate_posterior=variational_posteriors,
                optimizer=optimizer,
                num_steps=num_variational_steps)
            return elbo_loss_curve
        elbo_loss_curve = train()

        self.posterior=variational_posteriors
        return  elbo_loss_curve



def plot_forecast(self, forecast_mean, forecast_scale):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    dates = self.ts.index
    dates = list(range(len(self.ts.index)))
    ax.plot(dates[-self.num_forecast_steps:], self.ts[-self.num_forecast_steps:], label="true")
    forecast_steps = dates[-self.num_forecast_steps:]
    ax.plot(forecast_steps, forecast_mean, label="Forecast")
    # ax.plot(forecast_steps, forecast_samples.T, lw=1, alpha=0.1);
    ax.axvline(dates[-self.num_forecast_steps], linestyle="--")
    ax.fill_between(forecast_steps,
                    forecast_mean - 2 * forecast_scale,
                    forecast_mean + 2 * forecast_scale, alpha=0.2, label=r"$\pm 2\sigma$")
    ax.legend()



if __name__ == '__main__':
    file: str = "data/agh_core30min.csv"
    df = pd.read_csv(file, index_col='timestamp')
    df.index = pd.to_datetime(df.index)
    df.sort_values(by=['timestamp'], ascending=True)
    ts = df['cyfronet-ucirtr']

    predictor = TimeSeriesPredictor(num_forecast_steps=2*2*24, ts=ts)
    elbo=predictor.fit(num_variational_steps=300)
    plt.plot(elbo)
    plt.savefig('ssm/fit_orig.svg')
    predictor.save()

    forecast_mean, forecast_scale = predictor.predict()
    plot_forecast(predictor, forecast_mean, forecast_scale)
    plt.savefig('ssm/predict_orig.svg')


    file: str = "data/agh_core2h.csv"
    df = pd.read_csv(file, index_col='timestamp')
    df.index = pd.to_datetime(df.index)
    df.sort_values(by=['timestamp'], ascending=True)
    ts2 = df['cyfronet-ucirtr']

    p = AggregatedPredictor(num_forecast_steps=2*2*24, ts=ts.astype(np.float32), ts2=ts2.astype(np.float32))
    plt.figure()
    elbo=p.fit(num_variational_steps=300)
    plt.plot(elbo)
    plt.savefig('ssm/fit_agg.svg')
    forecast_mean, forecast_scale = p.predict()
    plot_forecast(predictor, forecast_mean, forecast_scale)
    plt.savefig('ssm/predict_agg.svg')
    p.save()

